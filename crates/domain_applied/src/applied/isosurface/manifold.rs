//! Watertight volumetric grid sampling and manifold topology diagnostics.
//!
//! Marching Cubes produces open boundary edges whenever a negative scalar region
//! touches the outer faces of a [`VoxelGrid`]. This module embeds an
//! `nx * ny * nz` interior lattice inside a `(nx + 2) * (ny + 2) * (nz + 2)`
//! padded volume, supports both solid SDF (`f(x, y, z) - c`) and finite-thickness
//! sheet (`|f(x, y, z) - c| - t / 2`) offsetting, intersects interior samples with
//! the domain bounding-box SDF, and populates the 1-voxel exterior halo with
//! strictly positive distances so `extract_isosurface(&grid, 0.0)` always yields
//! a closed 2-manifold (`0` unpaired boundary half-edges).

use super::types::{Mesh, Point3D, VoxelGrid};
use std::collections::{HashMap, HashSet};

/// Minimum supported interior voxel resolution along each spatial axis.
pub const MIN_INTERIOR_DIM: usize = 2;

/// Maximum synchronous interior voxel resolution per axis (`64^3` interior, `66^3` padded).
pub const MAX_SYNC_INTERIOR_DIM: usize = 64;

/// Threshold-avoidance epsilon (`> Point3D::EPSILON`) preventing exact-corner vertex collapse.
const ZERO_THRESHOLD_NUDGE: f32 = 2.5e-6;

/// Implicit field offsetting mode used when constructing a watertight volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IsosurfaceMode {
    /// Solid sublevel set $\{ (x, y, z) \mid f(x, y, z) - c \le 0 \}$.
    Solid,
    /// Thickened zero-level shell $\{ (x, y, z) \mid |f(x, y, z) - c| - t/2 \le 0 \}$.
    Sheet {
        /// Full physical shell thickness $t > 0$.
        thickness: f64,
    },
}

/// Configuration for sampling a bounded, padded watertight [`VoxelGrid`].
#[derive(Debug, Clone, PartialEq)]
pub struct WatertightGridConfig {
    /// Lower `[x_min, y_min, z_min]` physical domain corner.
    pub min_bounds: [f64; 3],
    /// Upper `[x_max, y_max, z_max]` physical domain corner.
    pub max_bounds: [f64; 3],
    /// Interior sample counts `(nx, ny, nz)` prior to `+2` boundary padding.
    pub interior_dims: (usize, usize, usize),
    /// Scalar isovalue $c$ subtracted from $f(x, y, z)$.
    pub iso_level: f64,
    /// Solid or thickened-sheet evaluation mode.
    pub mode: IsosurfaceMode,
}

impl WatertightGridConfig {
    /// Builds a symmetric cubic `[-half_extent, half_extent]^3` configuration.
    #[must_use]
    #[verified_engine::verified]
    pub fn cubic(
        half_extent: f64,
        resolution: usize,
        iso_level: f64,
        mode: IsosurfaceMode,
    ) -> Self {
        let bound = half_extent.abs();
        Self {
            min_bounds: [-bound, -bound, -bound],
            max_bounds: [bound, bound, bound],
            interior_dims: (resolution, resolution, resolution),
            iso_level,
            mode,
        }
    }
}

/// Validation or field-evaluation errors returned by [`sample_watertight_grid`].
#[derive(Debug, Clone, PartialEq)]
pub enum WatertightGridError {
    /// Interior dimensions fall outside `MIN_INTERIOR_DIM..=MAX_SYNC_INTERIOR_DIM`.
    InvalidResolution {
        /// Requested interior `(nx, ny, nz)` dimensions.
        dims: (usize, usize, usize),
        /// Maximum permitted interior resolution per axis.
        max_dim: usize,
    },
    /// Domain bounds are non-finite or inverted (`min >= max`).
    InvalidBounds {
        /// Lower bounds provided.
        min_bounds: [f64; 3],
        /// Upper bounds provided.
        max_bounds: [f64; 3],
    },
    /// Iso-level $c$ is `NaN` or infinite.
    InvalidIsoLevel(f64),
    /// Sheet thickness $t$ is non-finite or $\le 0$.
    InvalidThickness(f64),
    /// Scalar field evaluated to `NaN` or $\pm\infty$ at interior voxel `(ix, iy, iz)`.
    NonFiniteSample {
        /// Zero-based interior X index.
        ix: usize,
        /// Zero-based interior Y index.
        iy: usize,
        /// Zero-based interior Z index.
        iz: usize,
    },
    /// Underlying [`VoxelGrid`] builder error.
    GridBuild(String),
}

impl std::fmt::Display for WatertightGridError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResolution { dims, max_dim } => write!(
                f,
                "Interior grid resolution {:?} must be between {} and {} per axis",
                dims, MIN_INTERIOR_DIM, max_dim
            ),
            Self::InvalidBounds {
                min_bounds,
                max_bounds,
            } => write!(
                f,
                "Domain bounds {:?}..{:?} must be finite with min < max on every axis",
                min_bounds, max_bounds
            ),
            Self::InvalidIsoLevel(iso) => write!(f, "Iso-level must be finite (got {iso})"),
            Self::InvalidThickness(t) => {
                write!(f, "Sheet thickness must be finite and > 0 (got {t})")
            }
            Self::NonFiniteSample { ix, iy, iz } => write!(
                f,
                "Scalar field produced a non-finite value at interior voxel ({ix}, {iy}, {iz})"
            ),
            Self::GridBuild(msg) => write!(f, "Failed to build padded voxel grid: {msg}"),
        }
    }
}

impl std::error::Error for WatertightGridError {}

/// Topological 2-manifold summary for an extracted triangle [`Mesh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ManifoldStats {
    /// Number of vertices in the mesh.
    pub vertex_count: usize,
    /// Number of triangles (`indices.len() / 3`).
    pub triangle_count: usize,
    /// Number of unique undirected edges `{u, v}`.
    pub undirected_edges: usize,
    /// Number of directed half-edges missing an opposite twin `(v, u)`.
    pub open_boundary_edges: usize,
    /// Number of directed half-edges appearing more than once.
    pub non_manifold_edges: usize,
    /// Number of topologically or geometrically degenerate triangles.
    pub degenerate_triangles: usize,
}

impl ManifoldStats {
    /// Returns `true` when the mesh is non-empty, has `0` open or duplicate half-edges,
    /// `0` degenerate triangles, and satisfies $3 F = 2 E$.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_watertight_two_manifold(&self) -> bool {
        self.triangle_count > 0
            && self.open_boundary_edges == 0
            && self.non_manifold_edges == 0
            && self.degenerate_triangles == 0
            && 3 * self.triangle_count == 2 * self.undirected_edges
    }
}

#[verified_engine::verified]
fn validate_config(config: &WatertightGridConfig) -> Result<[f64; 3], WatertightGridError> {
    let (nx, ny, nz) = config.interior_dims;
    let dims = [nx, ny, nz];
    if dims
        .iter()
        .any(|&d| !(MIN_INTERIOR_DIM..=MAX_SYNC_INTERIOR_DIM).contains(&d))
    {
        return Err(WatertightGridError::InvalidResolution {
            dims: config.interior_dims,
            max_dim: MAX_SYNC_INTERIOR_DIM,
        });
    }
    if !config.iso_level.is_finite() {
        return Err(WatertightGridError::InvalidIsoLevel(config.iso_level));
    }
    if let IsosurfaceMode::Sheet { thickness } = config.mode
        && (!thickness.is_finite() || thickness <= 0.0)
    {
        return Err(WatertightGridError::InvalidThickness(thickness));
    }

    let mut steps = [0.0_f64; 3];
    for axis in 0..3 {
        let lo = config.min_bounds[axis];
        let hi = config.max_bounds[axis];
        let span = hi - lo;
        if !lo.is_finite()
            || !hi.is_finite()
            || !span.is_finite()
            || span <= f64::from(Point3D::EPSILON) * 4.0
        {
            return Err(WatertightGridError::InvalidBounds {
                min_bounds: config.min_bounds,
                max_bounds: config.max_bounds,
            });
        }
        steps[axis] = span / dims[axis] as f64;
    }
    Ok(steps)
}

#[inline]
#[verified_engine::verified]
fn box_signed_distance(pos: [f64; 3], min_b: [f64; 3], max_b: [f64; 3]) -> f64 {
    let dx = (min_b[0] - pos[0]).max(pos[0] - max_b[0]);
    let dy = (min_b[1] - pos[1]).max(pos[1] - max_b[1]);
    let dz = (min_b[2] - pos[2]).max(pos[2] - max_b[2]);
    dx.max(dy).max(dz)
}

#[inline]
#[verified_engine::verified]
fn nudge_away_from_zero(val: f32) -> f32 {
    if val.abs() < ZERO_THRESHOLD_NUDGE {
        if val < 0.0 {
            -ZERO_THRESHOLD_NUDGE
        } else {
            ZERO_THRESHOLD_NUDGE
        }
    } else {
        val
    }
}

/// Samples a scalar field onto a `(nx + 2) * (ny + 2) * (nz + 2)` padded [`VoxelGrid`]
/// ready for watertight extraction at threshold `0.0`.
///
/// # Examples
///
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::isosurface::{
///     IsosurfaceMode, WatertightGridConfig, analyze_mesh_manifold, extract_isosurface,
///     sample_watertight_grid,
/// };
///
/// let cfg = WatertightGridConfig::cubic(1.5, 16, 0.0, IsosurfaceMode::Solid);
/// let grid = sample_watertight_grid(&cfg, |x, y, z| x * x + y * y + z * z - 1.0).unwrap();
/// let mesh = extract_isosurface(&grid, 0.0).unwrap();
/// let stats = analyze_mesh_manifold(&mesh);
/// assert!(stats.is_watertight_two_manifold());
/// ```
#[verified_engine::verified]
pub fn sample_watertight_grid<F>(
    config: &WatertightGridConfig,
    mut field: F,
) -> Result<VoxelGrid, WatertightGridError>
where
    F: FnMut(f64, f64, f64) -> f64,
{
    let steps = validate_config(config)?;
    let (nx, ny, nz) = config.interior_dims;
    let (pw, ph, pd) = (nx + 2, ny + 2, nz + 2);
    let mut data = Vec::with_capacity(pw * ph * pd);

    for pz in 0..pd {
        let z = config.min_bounds[2] + (pz as f64 - 0.5) * steps[2];
        for py in 0..ph {
            let y = config.min_bounds[1] + (py as f64 - 0.5) * steps[1];
            for px in 0..pw {
                let x = config.min_bounds[0] + (px as f64 - 0.5) * steps[0];
                let box_sdf = box_signed_distance([x, y, z], config.min_bounds, config.max_bounds);
                let is_halo = px == 0 || px > nx || py == 0 || py > ny || pz == 0 || pz > nz;
                let sample = if is_halo {
                    box_sdf.max(f64::from(ZERO_THRESHOLD_NUDGE) * 4.0) as f32
                } else {
                    let raw = field(x, y, z);
                    if !raw.is_finite() {
                        return Err(WatertightGridError::NonFiniteSample {
                            ix: px - 1,
                            iy: py - 1,
                            iz: pz - 1,
                        });
                    }
                    let shifted = raw - config.iso_level;
                    let mode_val = match config.mode {
                        IsosurfaceMode::Solid => shifted,
                        IsosurfaceMode::Sheet { thickness } => shifted.abs() - 0.5 * thickness,
                    };
                    nudge_away_from_zero(mode_val.max(box_sdf) as f32)
                };
                data.push(sample);
            }
        }
    }

    let origin = Point3D::new(
        (config.min_bounds[0] - 0.5 * steps[0]) as f32,
        (config.min_bounds[1] - 0.5 * steps[1]) as f32,
        (config.min_bounds[2] - 0.5 * steps[2]) as f32,
    );
    let voxel_size = Point3D::new(steps[0] as f32, steps[1] as f32, steps[2] as f32);

    VoxelGrid::builder()
        .dimensions(pw, ph, pd)
        .data(data)
        .voxel_size(voxel_size)
        .origin(origin)
        .build()
        .map_err(|err| WatertightGridError::GridBuild(err.to_string()))
}

#[inline]
#[verified_engine::verified]
fn is_triangle_degenerate(mesh: &Mesh, tri: [usize; 3]) -> bool {
    let n_verts = mesh.vertices.len();
    let [a, b, c] = tri;
    if a >= n_verts || b >= n_verts || c >= n_verts || a == b || b == c || c == a {
        return true;
    }
    let va = mesh.vertices[a];
    let vb = mesh.vertices[b];
    let vc = mesh.vertices[c];
    let cross = (vb - va).cross(vc - va);
    let area_sq = cross.dot(cross);
    !area_sq.is_finite() || area_sq <= 1.0e-24
}

/// Computes half-edge pairing and triangle-quality statistics for `mesh`.
#[must_use]
#[verified_engine::verified]
pub fn analyze_mesh_manifold(mesh: &Mesh) -> ManifoldStats {
    let triangle_count = mesh.indices.len() / 3;
    let mut directed_edges: HashMap<(usize, usize), usize> =
        HashMap::with_capacity(triangle_count * 3);
    let mut undirected: HashSet<(usize, usize)> = HashSet::with_capacity(triangle_count * 2);
    let mut degenerate_triangles = usize::from(!mesh.indices.len().is_multiple_of(3));

    for &[a, b, c] in mesh.indices.as_chunks::<3>().0 {
        if is_triangle_degenerate(mesh, [a, b, c]) {
            degenerate_triangles += 1;
        }
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *directed_edges.entry((u, v)).or_insert(0) += 1;
            let canonical = if u <= v { (u, v) } else { (v, u) };
            undirected.insert(canonical);
        }
    }

    let mut open_boundary_edges = 0_usize;
    let mut non_manifold_edges = 0_usize;
    for (&(u, v), &count) in &directed_edges {
        if count > 1 {
            non_manifold_edges += count - 1;
        }
        if directed_edges.get(&(v, u)).copied().unwrap_or(0) != 1 {
            open_boundary_edges += 1;
        }
    }

    ManifoldStats {
        vertex_count: mesh.vertices.len(),
        triangle_count,
        undirected_edges: undirected.len(),
        open_boundary_edges,
        non_manifold_edges,
        degenerate_triangles,
    }
}
