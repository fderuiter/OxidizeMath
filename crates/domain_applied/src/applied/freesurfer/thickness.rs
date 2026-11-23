//! Uniform CSR spatial grid (`FlatSpatialGrid`), preallocated `ThicknessWorkspace`, and
//! zero-allocation single-vertex and batch cortical thickness kernels.

use super::surface::Surface;
use math_commons::math_kernel::types::flatten_3d_index;

#[verified_engine::verified]
fn grid_axis_dim(vertex_count: usize) -> usize {
    if vertex_count == 0 {
        return 0;
    }
    let mut m = ((vertex_count as f64).cbrt() as usize).max(1);
    while m > 1 && (m - 1).saturating_mul(m - 1).saturating_mul(m - 1) >= vertex_count {
        m -= 1;
    }
    while m.saturating_mul(m).saturating_mul(m) < vertex_count {
        m += 1;
    }
    m
}

#[verified_engine::verified]
fn compute_surface_bounds(vertices: &[[f64; 3]], m: usize) -> ([f64; 3], f64) {
    let mut min_b = [f64::INFINITY; 3];
    let mut max_b = [f64::NEG_INFINITY; 3];
    for v in vertices {
        if v[0].is_finite() && v[1].is_finite() && v[2].is_finite() {
            for axis in 0..3 {
                min_b[axis] = min_b[axis].min(v[axis]);
                max_b[axis] = max_b[axis].max(v[axis]);
            }
        }
    }
    if !min_b[0].is_finite() {
        return ([0.0; 3], 1.0);
    }
    let max_extent = (max_b[0] - min_b[0])
        .max(max_b[1] - min_b[1])
        .max(max_b[2] - min_b[2]);
    let cell_size = if max_extent.is_finite() && max_extent > 0.0 {
        (max_extent / (m as f64)).max(1e-6)
    } else {
        1.0
    };
    (min_b, cell_size)
}

#[inline]
#[verified_engine::verified]
fn coord_to_cell(coord: f64, origin: f64, cell_size: f64, m: usize) -> usize {
    if m <= 1 || !coord.is_finite() {
        return if coord.is_sign_positive() && !coord.is_nan() && m > 1 {
            m - 1
        } else {
            0
        };
    }
    let scaled = ((coord - origin) / cell_size).floor();
    let mut idx = if scaled <= 0.0 {
        0
    } else if scaled >= (m - 1) as f64 {
        m - 1
    } else {
        scaled as usize
    };
    if idx + 1 < m && coord >= origin + ((idx + 1) as f64) * cell_size {
        idx += 1;
    } else if idx > 0 && coord < origin + (idx as f64) * cell_size {
        idx -= 1;
    }
    idx
}

#[inline]
#[verified_engine::verified]
fn vertex_cell_index(v: &[f64; 3], origin: &[f64; 3], cell_size: f64, m: usize) -> usize {
    let ix = coord_to_cell(v[0], origin[0], cell_size, m);
    let iy = coord_to_cell(v[1], origin[1], cell_size, m);
    let iz = coord_to_cell(v[2], origin[2], cell_size, m);
    flatten_3d_index(ix, iy, iz, m, m)
}

/// Core-only compressed-sparse-row (CSR) uniform 3D voxel grid over preallocated slices.
///
/// Partitions $V$ surface vertices into an $[0, M - 1]^3$ grid ($M = \lceil V^{1/3} \rceil$) using
/// counting-sort `cell_offsets` and `sorted_vertices` slices without heap allocation.
#[derive(Clone, Copy, Debug)]
pub struct FlatSpatialGrid<'a> {
    origin: [f64; 3],
    cell_size: f64,
    grid_dim: usize,
    cell_offsets: &'a [usize],
    sorted_vertices: &'a [[f64; 3]],
}

impl<'a> FlatSpatialGrid<'a> {
    /// Returns the minimum `cell_offsets` slice length ($M^3 + 1$, or `0` when `vertex_count == 0`).
    #[must_use]
    #[verified_engine::verified]
    pub fn required_offsets_len(vertex_count: usize) -> usize {
        if vertex_count == 0 {
            return 0;
        }
        let m = grid_axis_dim(vertex_count);
        m.saturating_mul(m).saturating_mul(m).saturating_add(1)
    }

    /// Constructs a CSR [`FlatSpatialGrid`] partitioning `surface.vertices` into caller-provided slices.
    ///
    /// Returns `None` when `cell_offsets` or `sorted_vertices` lacks capacity.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_surface(
        surface: &Surface,
        cell_offsets: &'a mut [usize],
        sorted_vertices: &'a mut [[f64; 3]],
    ) -> Option<Self> {
        let vertices = surface.vertices.as_slice();
        let n = vertices.len();
        if n == 0 {
            return Some(Self {
                origin: [0.0; 3],
                cell_size: 1.0,
                grid_dim: 0,
                cell_offsets: &[],
                sorted_vertices: &[],
            });
        }
        let m = grid_axis_dim(n);
        let num_cells = m.checked_mul(m)?.checked_mul(m)?;
        let offsets_len = num_cells.checked_add(1)?;
        if cell_offsets.len() < offsets_len || sorted_vertices.len() < n {
            return None;
        }

        let (origin, cell_size) = compute_surface_bounds(vertices, m);
        let offsets = &mut cell_offsets[..offsets_len];
        let sorted = &mut sorted_vertices[..n];
        offsets.fill(0);

        for v in vertices {
            offsets[vertex_cell_index(v, &origin, cell_size, m)] += 1;
        }

        let mut running = 0_usize;
        for slot in offsets.iter_mut() {
            let count = *slot;
            *slot = running;
            running += count;
        }

        for &v in vertices {
            let cell = vertex_cell_index(&v, &origin, cell_size, m);
            let dst = offsets[cell];
            sorted[dst] = v;
            offsets[cell] = dst + 1;
        }

        offsets.copy_within(0..num_cells, 1);
        offsets[0] = 0;

        Some(Self {
            origin,
            cell_size,
            grid_dim: m,
            cell_offsets: offsets,
            sorted_vertices: sorted,
        })
    }

    /// Returns `true` when the indexed surface contains zero vertices.
    #[must_use]
    #[verified_engine::verified]
    pub const fn is_empty(&self) -> bool {
        self.sorted_vertices.is_empty()
    }

    /// Returns the isotropic cubic voxel spacing (`1.0` on empty or zero-extent bounds).
    #[must_use]
    #[verified_engine::verified]
    pub const fn cell_size(&self) -> f64 {
        self.cell_size
    }

    #[verified_engine::verified]
    fn scan_shell(
        &self,
        point: &[f64; 3],
        center: [usize; 3],
        radius: usize,
        mut min_sq: f64,
    ) -> f64 {
        let m = self.grid_dim;
        let [cx, cy, cz] = center;
        let (lo_x, hi_x) = (cx.saturating_sub(radius), (cx + radius).min(m - 1));
        let (lo_y, hi_y) = (cy.saturating_sub(radius), (cy + radius).min(m - 1));
        let (lo_z, hi_z) = (cz.saturating_sub(radius), (cz + radius).min(m - 1));

        for iz in lo_z..=hi_z {
            let z_on_shell = iz.abs_diff(cz) == radius;
            for iy in lo_y..=hi_y {
                let yz_on_shell = z_on_shell || iy.abs_diff(cy) == radius;
                for ix in lo_x..=hi_x {
                    if !yz_on_shell && ix.abs_diff(cx) < radius {
                        continue;
                    }
                    let cell = flatten_3d_index(ix, iy, iz, m, m);
                    for v in
                        &self.sorted_vertices[self.cell_offsets[cell]..self.cell_offsets[cell + 1]]
                    {
                        let dx = point[0] - v[0];
                        let dy = point[1] - v[1];
                        let dz = point[2] - v[2];
                        let d_sq = dx * dx + dy * dy + dz * dz;
                        if d_sq < min_sq {
                            min_sq = d_sq;
                        }
                    }
                }
            }
        }
        min_sq
    }

    #[verified_engine::verified]
    fn unsearched_boundary_distance(
        &self,
        point: &[f64; 3],
        center: [usize; 3],
        radius: usize,
    ) -> f64 {
        let m = self.grid_dim;
        let mut bound = f64::INFINITY;
        for axis in 0..3 {
            let lo = center[axis].saturating_sub(radius);
            let hi = (center[axis] + radius).min(m - 1);
            if lo > 0 {
                bound = bound.min(point[axis] - (self.origin[axis] + (lo as f64) * self.cell_size));
            }
            if hi + 1 < m {
                bound = bound
                    .min((self.origin[axis] + ((hi + 1) as f64) * self.cell_size) - point[axis]);
            }
        }
        bound
    }

    /// Queries the exact shortest Euclidean distance from `point` to any indexed vertex.
    ///
    /// Clamps exterior query coordinates and concentric Chebyshev shells to `[0, M - 1]^3`,
    /// returning `f64::INFINITY` on empty grids and `f64::NAN` on `NaN` query coordinates.
    #[must_use]
    #[verified_engine::verified]
    pub fn shortest_distance(&self, point: &[f64; 3]) -> f64 {
        if self.sorted_vertices.is_empty() {
            return f64::INFINITY;
        }
        if point[0].is_nan() || point[1].is_nan() || point[2].is_nan() {
            return f64::NAN;
        }

        let m = self.grid_dim;
        let center = [
            coord_to_cell(point[0], self.origin[0], self.cell_size, m),
            coord_to_cell(point[1], self.origin[1], self.cell_size, m),
            coord_to_cell(point[2], self.origin[2], self.cell_size, m),
        ];

        let mut min_dist_sq = f64::INFINITY;
        for radius in 0..m {
            min_dist_sq = self.scan_shell(point, center, radius, min_dist_sq);
            let min_distance = min_dist_sq.sqrt();
            if min_distance <= self.unsearched_boundary_distance(point, center, radius) {
                return min_distance;
            }
        }
        min_dist_sq.sqrt()
    }
}

/// Reusable CSR buffer workspace enabling zero-allocation [`FlatSpatialGrid`] indexing and
/// full-mesh [`cortical_thickness_map_into`] evaluation inside `VerifiedAllocator`-locked frames.
#[derive(Clone, Debug, Default)]
pub struct ThicknessWorkspace {
    white_offsets: Vec<usize>,
    white_vertices: Vec<[f64; 3]>,
    pial_offsets: Vec<usize>,
    pial_vertices: Vec<[f64; 3]>,
}

impl ThicknessWorkspace {
    /// Creates an empty unallocated [`ThicknessWorkspace`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            white_offsets: Vec::new(),
            white_vertices: Vec::new(),
            pial_offsets: Vec::new(),
            pial_vertices: Vec::new(),
        }
    }

    /// Preallocates CSR offset and vertex buffers for surfaces with up to `max_vertices` vertices.
    #[must_use]
    pub fn with_max_vertices(max_vertices: usize) -> Self {
        let offsets_len = FlatSpatialGrid::required_offsets_len(max_vertices);
        Self {
            white_offsets: vec![0; offsets_len],
            white_vertices: vec![[0.0; 3]; max_vertices],
            pial_offsets: vec![0; offsets_len],
            pial_vertices: vec![[0.0; 3]; max_vertices],
        }
    }

    /// Grows internal CSR buffers if needed to accommodate at least `max_vertices` per surface.
    pub fn ensure_capacity(&mut self, max_vertices: usize) {
        let offsets_len = FlatSpatialGrid::required_offsets_len(max_vertices);
        if self.white_offsets.len() < offsets_len {
            self.white_offsets.resize(offsets_len, 0);
        }
        if self.white_vertices.len() < max_vertices {
            self.white_vertices.resize(max_vertices, [0.0; 3]);
        }
        if self.pial_offsets.len() < offsets_len {
            self.pial_offsets.resize(offsets_len, 0);
        }
        if self.pial_vertices.len() < max_vertices {
            self.pial_vertices.resize(max_vertices, [0.0; 3]);
        }
    }

    /// Builds `(white_grid, pial_grid)` in-place without heap allocation.
    ///
    /// Returns `None` when preallocated workspace slices lack capacity for either surface.
    #[must_use]
    #[verified_engine::verified]
    pub fn build_grids<'a>(
        &'a mut self,
        white_surface: &Surface,
        pial_surface: &Surface,
    ) -> Option<(FlatSpatialGrid<'a>, FlatSpatialGrid<'a>)> {
        let white_grid = FlatSpatialGrid::from_surface(
            white_surface,
            &mut self.white_offsets,
            &mut self.white_vertices,
        )?;
        let pial_grid = FlatSpatialGrid::from_surface(
            pial_surface,
            &mut self.pial_offsets,
            &mut self.pial_vertices,
        )?;
        Some((white_grid, pial_grid))
    }
}

/// Calculates the shortest Euclidean distance from `point` to `surface` via a direct scan.
#[verified_engine::verified]
fn point_to_surface_distance(point: &[f64; 3], surface: &Surface) -> f64 {
    if surface.vertices.is_empty() {
        return f64::INFINITY;
    }
    if point[0].is_nan() || point[1].is_nan() || point[2].is_nan() {
        return f64::NAN;
    }

    let mut min_dist_sq = f64::INFINITY;
    for v in &surface.vertices {
        let dx = point[0] - v[0];
        let dy = point[1] - v[1];
        let dz = point[2] - v[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
        }
    }
    min_dist_sq.sqrt()
}

/// Calculates symmetric cortical thickness at `(v_w, v_p)` via direct vertex scan.
#[verified_engine::verified]
pub fn cortical_thickness(
    v_w: &[f64; 3],
    v_p: &[f64; 3],
    white_surface: &Surface,
    pial_surface: &Surface,
) -> f64 {
    let dist_w_to_p = point_to_surface_distance(v_w, pial_surface);
    let dist_p_to_w = point_to_surface_distance(v_p, white_surface);
    0.5 * (dist_w_to_p + dist_p_to_w)
}

/// Evaluates symmetric cortical thickness at `(v_w, v_p)` in expected $O(1)$ time using
/// prebuilt [`FlatSpatialGrid`] indices without dynamic heap allocation.
#[must_use]
#[verified_engine::verified]
pub fn cortical_thickness_indexed(
    v_w: &[f64; 3],
    v_p: &[f64; 3],
    white_grid: &FlatSpatialGrid<'_>,
    pial_grid: &FlatSpatialGrid<'_>,
) -> f64 {
    let dist_w_to_p = pial_grid.shortest_distance(v_w);
    let dist_p_to_w = white_grid.shortest_distance(v_p);
    0.5 * (dist_w_to_p + dist_p_to_w)
}

/// Computes symmetric cortical thickness across all corresponding `(white, pial)` vertex pairs
/// into `thicknesses_out` using `workspace` with zero dynamic heap allocations.
///
/// Returns the number of vertex thicknesses written, or `0` when either surface is empty or when
/// `workspace` or `thicknesses_out` lacks capacity.
#[verified_engine::verified]
pub fn cortical_thickness_map_into(
    white_surface: &Surface,
    pial_surface: &Surface,
    workspace: &mut ThicknessWorkspace,
    thicknesses_out: &mut [f64],
) -> usize {
    let count = white_surface
        .vertices
        .len()
        .min(pial_surface.vertices.len());
    if count == 0 || thicknesses_out.len() < count {
        return 0;
    }
    let Some((white_grid, pial_grid)) = workspace.build_grids(white_surface, pial_surface) else {
        return 0;
    };
    for (idx, out) in thicknesses_out.iter_mut().enumerate().take(count) {
        *out = cortical_thickness_indexed(
            &white_surface.vertices[idx],
            &pial_surface.vertices[idx],
            &white_grid,
            &pial_grid,
        );
    }
    count
}

/// Computes symmetric cortical thickness across all corresponding `(white, pial)` vertex pairs,
/// constructing a uniform CSR [`FlatSpatialGrid`] once per surface mesh.
#[must_use]
pub fn cortical_thickness_map(white_surface: &Surface, pial_surface: &Surface) -> Vec<f64> {
    let count = white_surface
        .vertices
        .len()
        .min(pial_surface.vertices.len());
    if count == 0 {
        return Vec::new();
    }
    let max_verts = white_surface
        .vertices
        .len()
        .max(pial_surface.vertices.len());
    let mut workspace = ThicknessWorkspace::with_max_vertices(max_verts);
    let mut thicknesses = vec![0.0; count];
    let written = cortical_thickness_map_into(
        white_surface,
        pial_surface,
        &mut workspace,
        &mut thicknesses,
    );
    thicknesses.truncate(written);
    thicknesses
}

/// Evaluates symmetric white-to-pial cortical thickness across paired vertices into `out`.
pub fn cortical_thickness_batch_into(
    white_surface: &Surface,
    pial_surface: &Surface,
    out: &mut [f64],
) {
    let (nw, np) = (white_surface.vertices.len(), pial_surface.vertices.len());
    let count = nw.min(np).min(out.len());
    if count > 0 {
        let mut ws = ThicknessWorkspace::with_max_vertices(nw.max(np));
        let _ =
            cortical_thickness_map_into(white_surface, pial_surface, &mut ws, &mut out[..count]);
    }
}

/// Evaluates symmetric white-to-pial cortical thickness across all paired vertices via [`cortical_thickness_map`].
#[must_use]
pub fn cortical_thickness_batch(white_surface: &Surface, pial_surface: &Surface) -> Vec<f64> {
    cortical_thickness_map(white_surface, pial_surface)
}

#[cfg(test)]
#[path = "thickness_tests.rs"]
mod tests;
