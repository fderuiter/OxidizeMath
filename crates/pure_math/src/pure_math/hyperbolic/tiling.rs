// [cite:pure_math]
use super::{GeodesicDisk, HalfPlanePoint, HyperbolicError, Psu11, cayley_half_plane_to_disk};
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

/// Maximum BFS reflection depth allowed for hyperbolic tilings.
pub const MAX_TILING_DEPTH: usize = 5;

/// Hard upper bound on generated polygons per tiling buffer.
pub const MAX_TILING_POLYGONS: usize = 512;

/// Maximum inline vertex count supported by [`TiledPolygon`].
pub const MAX_POLYGON_VERTICES: usize = 16;

/// Number of finite horocyclic vertices used per truncated `{infinity, q}` apeirogon.
const IDEAL_APEIROGON_FINITE_VERTICES: usize = 14;

/// Hyperbolic Schläfli tiling specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TilingKind {
    /// Regular compact `{p, q}` tiling with $(p - 2)(q - 2) > 4$.
    Regular {
        /// Number of geodesic sides per polygon ($3 \le p \le 16$).
        p: u32,
        /// Number of polygons meeting at each vertex ($q \ge 3$).
        q: u32,
    },
    /// Ideal apeirogonal `{infinity, q}` Hecke tiling ($3 \le q \le 16$).
    Ideal {
        /// Number of apeirogons meeting at each finite vertex ($q \ge 3$).
        q: u32,
    },
}

/// Fixed-capacity hyperbolic tile stored inline without per-polygon heap allocations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TiledPolygon {
    /// Polygon vertices in $\overline{\mathbb{D}}$ (`0..vertex_count` active).
    pub vertices: [Complex64; MAX_POLYGON_VERTICES],
    /// Active vertex count (`3..=MAX_POLYGON_VERTICES`).
    pub vertex_count: usize,
    /// Reflection generation depth (`0..=MAX_TILING_DEPTH`).
    pub depth: u8,
    /// Canonical deduplication anchor (tile center for `{p,q}`, ideal cusp for `{infinity,q}`).
    pub anchor: Complex64,
}

impl Default for TiledPolygon {
    #[verified_engine::verified]
    fn default() -> Self {
        let poly = Self {
            vertices: [Complex64::new(0.0, 0.0); MAX_POLYGON_VERTICES],
            vertex_count: 0,
            depth: 0,
            anchor: Complex64::new(0.0, 0.0),
        };
        debug_assert_eq!(poly.vertex_count, 0);
        poly
    }
}

impl TiledPolygon {
    /// Returns the active vertices slice `&self.vertices[..self.vertex_count]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn active_vertices(&self) -> &[Complex64] {
        assert!(self.vertex_count <= MAX_POLYGON_VERTICES);
        &self.vertices[..self.vertex_count]
    }
}

/// Validates that `{p, q}` satisfies the compact hyperbolic Schläfli condition $(p - 2)(q - 2) > 4$.
#[verified_engine::verified]
pub fn validate_hyperbolic_schlafli(p: u32, q: u32) -> Result<(), HyperbolicError> {
    if p < 3 || q < 3 {
        return Err(HyperbolicError::NonHyperbolicSchlafli { p, q });
    }
    let lhs = u64::from(p - 2) * u64::from(q - 2);
    if lhs <= 4 {
        return Err(HyperbolicError::NonHyperbolicSchlafli { p, q });
    }
    assert!(lhs > 4);
    Ok(())
}

/// Computes the exact Euclidean vertex radius $r_0 \in (0, 1)$ of a central regular `{p, q}` polygon in $\mathbb{D}$.
#[verified_engine::verified]
pub fn central_polygon_radius(p: u32, q: u32) -> Result<f64, HyperbolicError> {
    validate_hyperbolic_schlafli(p, q)?;
    let alpha = PI / f64::from(p);
    let beta = PI / f64::from(q);
    let num = (alpha + beta).cos();
    let den = (alpha - beta).cos();
    assert!(num > 0.0 && den > num);
    let r0 = (num / den).sqrt();
    assert!(r0 > 0.0 && r0 < 1.0);
    Ok(r0)
}

/// Generates the $p$ counterclockwise vertices of the central regular `{p, q}` polygon in $\mathbb{D}$.
#[verified_engine::verified]
pub fn regular_polygon_vertices(p: u32, q: u32) -> Result<Vec<Complex64>, HyperbolicError> {
    let r0 = central_polygon_radius(p, q)?;
    let n = p as usize;
    let mut verts = Vec::with_capacity(n);
    for k in 0..n {
        let theta = TAU * (k as f64) / (n as f64);
        verts.push(Complex64::new(r0 * theta.cos(), r0 * theta.sin()));
    }
    assert_eq!(verts.len(), n);
    Ok(verts)
}

/// Generates the $n$ counterclockwise ideal vertices $\xi_k = e^{2\pi i k / n} \in \partial\mathbb{D}$ of a regular ideal $n$-gon.
#[verified_engine::verified]
pub fn ideal_polygon_vertices(sides: u32) -> Result<Vec<Complex64>, HyperbolicError> {
    if sides < 3 || (sides as usize) > MAX_POLYGON_VERTICES {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "ideal polygon side count must lie in 3..=16",
        });
    }
    let n = sides as usize;
    let mut verts = Vec::with_capacity(n);
    for k in 0..n {
        let theta = TAU * (k as f64) / (n as f64);
        verts.push(Complex64::new(theta.cos(), theta.sin()));
    }
    assert_eq!(verts.len(), n);
    Ok(verts)
}

/// Builds the seed tile at depth 0 for a compact `{p, q}` tiling.
#[verified_engine::verified]
fn build_regular_seed_tile(p: u32, q: u32) -> Result<TiledPolygon, HyperbolicError> {
    if (p as usize) > MAX_POLYGON_VERTICES {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "tiling polygon side count exceeds MAX_POLYGON_VERTICES (16)",
        });
    }
    let r0 = central_polygon_radius(p, q)?;
    let n = p as usize;
    let mut tile = TiledPolygon {
        vertices: [Complex64::new(0.0, 0.0); MAX_POLYGON_VERTICES],
        vertex_count: n,
        depth: 0,
        anchor: Complex64::new(0.0, 0.0),
    };
    for (k, vertex) in tile.vertices[..n].iter_mut().enumerate() {
        let theta = TAU * (k as f64) / (n as f64);
        *vertex = Complex64::new(r0 * theta.cos(), r0 * theta.sin());
    }
    assert_eq!(tile.vertex_count, n);
    Ok(tile)
}

/// Builds the seed apeirogonal tile at depth 0 for an ideal `{infinity, q}` Hecke tiling.
#[verified_engine::verified]
fn build_ideal_apeirogon_seed_tile(q: u32) -> Result<TiledPolygon, HyperbolicError> {
    if q < 3 || (q as usize) > MAX_POLYGON_VERTICES {
        return Err(HyperbolicError::NonHyperbolicSchlafli { p: 0, q });
    }
    let alpha = PI / f64::from(q);
    let step = 2.0 * alpha.cos();
    let height = alpha.sin();
    let half_span = (IDEAL_APEIROGON_FINITE_VERTICES as i32) / 2;
    let mut tile = TiledPolygon {
        vertices: [Complex64::new(0.0, 0.0); MAX_POLYGON_VERTICES],
        vertex_count: IDEAL_APEIROGON_FINITE_VERTICES + 1,
        depth: 0,
        anchor: Complex64::new(1.0, 0.0),
    };
    for (idx, vertex) in tile.vertices[..IDEAL_APEIROGON_FINITE_VERTICES]
        .iter_mut()
        .enumerate()
    {
        let k = (idx as i32) - half_span + 1;
        let u = (f64::from(k) - 0.5) * step;
        let hp = HalfPlanePoint::new_clamped(u, height);
        *vertex = cayley_half_plane_to_disk(hp).as_complex();
    }
    tile.vertices[IDEAL_APEIROGON_FINITE_VERTICES] = Complex64::new(1.0, 0.0);
    assert_eq!(tile.vertex_count, IDEAL_APEIROGON_FINITE_VERTICES + 1);
    Ok(tile)
}

/// Reflects `parent` across the geodesic edge `(v_edge, v_next)` into a child tile at `depth`.
#[verified_engine::verified]
fn reflect_tile_across_edge(
    parent: &TiledPolygon,
    edge_idx: usize,
    next_depth: u8,
) -> Option<TiledPolygon> {
    assert!(edge_idx < parent.vertex_count);
    let v0 = parent.vertices[edge_idx];
    let v1 = parent.vertices[(edge_idx + 1) % parent.vertex_count];
    if v0.norm_sqr() >= 1.0 - 1.0e-9 || v1.norm_sqr() >= 1.0 - 1.0e-9 {
        return None;
    }
    let mirror = GeodesicDisk::through_points(v0, v1).ok()?.inversion();
    let mut child = TiledPolygon {
        vertices: [Complex64::new(0.0, 0.0); MAX_POLYGON_VERTICES],
        vertex_count: parent.vertex_count,
        depth: next_depth,
        anchor: mirror.reflect_complex(parent.anchor),
    };
    for i in 0..parent.vertex_count {
        child.vertices[i] = mirror.reflect_complex(parent.vertices[i]);
    }
    debug_assert_eq!(child.vertex_count, parent.vertex_count);
    Some(child)
}

/// Checks whether `candidate_anchor` matches any existing tile anchor in `tiles`.
#[verified_engine::verified]
fn contains_duplicate_anchor(
    tiles: &[TiledPolygon],
    candidate_anchor: Complex64,
    tol_sq: f64,
) -> bool {
    assert!(tol_sq > 0.0);
    for existing in tiles {
        if (existing.anchor - candidate_anchor).norm_sqr() <= tol_sq {
            return true;
        }
    }
    false
}

/// Pre-allocated hyperbolic tiling buffer supporting zero-allocation `PSU(1,1)` transformations.
#[derive(Clone, Debug)]
pub struct HyperbolicTilingBuffer {
    kind: TilingKind,
    depth: usize,
    polygons: Vec<TiledPolygon>,
}

impl HyperbolicTilingBuffer {
    /// Allocates a new [`HyperbolicTilingBuffer`] with capacity [`MAX_TILING_POLYGONS`] and populates `kind`.
    #[verified_engine::verified]
    pub fn new(kind: TilingKind, depth: usize) -> Result<Self, HyperbolicError> {
        let mut buf = Self {
            kind,
            depth: depth.min(MAX_TILING_DEPTH),
            polygons: Vec::with_capacity(MAX_TILING_POLYGONS),
        };
        buf.rebuild(kind, depth)?;
        assert!(!buf.polygons.is_empty() && buf.polygons.len() <= MAX_TILING_POLYGONS);
        Ok(buf)
    }

    /// Rebuilds the tiling in-place inside the pre-allocated polygon buffer (`depth <= 5`, `<= 512` tiles).
    #[verified_engine::verified]
    pub fn rebuild(&mut self, kind: TilingKind, depth: usize) -> Result<usize, HyperbolicError> {
        let capped_depth = depth.min(MAX_TILING_DEPTH);
        let (seed, dedup_tol_sq) = match kind {
            TilingKind::Regular { p, q } => (build_regular_seed_tile(p, q)?, 1.0e-7),
            TilingKind::Ideal { q } => (build_ideal_apeirogon_seed_tile(q)?, 1.0e-8),
        };
        self.kind = kind;
        self.depth = capped_depth;
        self.polygons.clear();
        self.polygons.push(seed);
        let mut cursor = 0usize;
        while cursor < self.polygons.len() && self.polygons.len() < MAX_TILING_POLYGONS {
            let parent = self.polygons[cursor];
            cursor += 1;
            if usize::from(parent.depth) >= capped_depth {
                continue;
            }
            let next_depth = parent.depth + 1;
            for edge_idx in 0..parent.vertex_count {
                if self.polygons.len() >= MAX_TILING_POLYGONS {
                    break;
                }
                if let Some(child) = reflect_tile_across_edge(&parent, edge_idx, next_depth)
                    && !contains_duplicate_anchor(&self.polygons, child.anchor, dedup_tol_sq)
                {
                    self.polygons.push(child);
                }
            }
        }
        assert!(!self.polygons.is_empty() && self.polygons.len() <= MAX_TILING_POLYGONS);
        Ok(self.polygons.len())
    }

    /// Returns the active [`TilingKind`].
    #[must_use]
    #[verified_engine::verified]
    pub fn kind(&self) -> TilingKind {
        debug_assert!(self.depth <= MAX_TILING_DEPTH);
        self.kind
    }

    /// Returns the capped BFS reflection depth (`<= 5`).
    #[must_use]
    #[verified_engine::verified]
    pub fn depth(&self) -> usize {
        assert!(self.depth <= MAX_TILING_DEPTH);
        self.depth
    }

    /// Returns the generated immutable base polygons in $\overline{\mathbb{D}}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn polygons(&self) -> &[TiledPolygon] {
        assert!(self.polygons.len() <= MAX_TILING_POLYGONS);
        &self.polygons
    }

    /// Applies `isometry` to every polygon into pre-allocated `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn transform_into(&self, isometry: Psu11, out: &mut [TiledPolygon]) -> usize {
        assert!((isometry.determinant() - 1.0).abs() < 1.0e-6);
        let count = self.polygons.len().min(out.len());
        for (src, dst) in self.polygons.iter().zip(out.iter_mut()).take(count) {
            dst.vertex_count = src.vertex_count;
            dst.depth = src.depth;
            dst.anchor = isometry.apply_complex(src.anchor);
            for (v_dst, &v_src) in dst
                .vertices
                .iter_mut()
                .zip(src.vertices.iter())
                .take(src.vertex_count)
            {
                *v_dst = isometry.apply_complex(v_src);
            }
        }
        assert!(count <= MAX_TILING_POLYGONS);
        count
    }
}
