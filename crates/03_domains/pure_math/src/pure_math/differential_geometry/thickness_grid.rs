//! CSR spatial uniform grids (`FlatSpatialGrid`, `SpatialGrid`) and `ThicknessWorkspace`.
// [cite:pure_math]

use super::Surface;
use math_commons::math_kernel::types::flatten_3d_index;
use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};

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

#[verified_engine::verified]
pub(super) fn linear_scan_distance(point: &[f64; 3], vertices: &[[f64; 3]]) -> f64 {
    if vertices.is_empty() {
        return f64::INFINITY;
    }
    if point[0].is_nan() || point[1].is_nan() || point[2].is_nan() {
        return f64::NAN;
    }
    let mut min_dist_sq = f64::INFINITY;
    for v in vertices {
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
        if !point[0].is_finite() || !point[1].is_finite() || !point[2].is_finite() {
            return linear_scan_distance(point, self.sorted_vertices);
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

/// Owned uniform 3D voxel grid backed by a contiguous CSR vertex buffer for reusable
/// zero-allocation distance queries across pre-indexed cortical surfaces.
///
/// # Example
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::differential_geometry::{SpatialGrid, Surface};
///
/// let surface = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
/// let grid = SpatialGrid::from_surface(&surface);
/// assert_eq!(grid.shortest_distance(&[0.0, 3.0, 4.0]), 5.0);
/// ```
#[derive(Clone, Debug)]
pub struct SpatialGrid {
    origin: [f64; 3],
    cell_size: f64,
    grid_dim: usize,
    cell_offsets: Vec<usize>,
    sorted_vertices: Vec<[f64; 3]>,
}

impl SpatialGrid {
    /// Constructs a reusable flat CSR spatial uniform grid partitioning `surface` vertices.
    #[must_use]
    pub fn from_surface(surface: &Surface) -> Self {
        let n = surface.vertices.len();
        let offsets_len = FlatSpatialGrid::required_offsets_len(n);
        let mut cell_offsets = vec![0; offsets_len];
        let mut sorted_vertices = vec![[0.0; 3]; n];
        let (origin, cell_size, grid_dim) =
            match FlatSpatialGrid::from_surface(surface, &mut cell_offsets, &mut sorted_vertices) {
                Some(grid) => (grid.origin, grid.cell_size, grid.grid_dim),
                None => ([0.0; 3], 1.0, 0),
            };
        Self {
            origin,
            cell_size,
            grid_dim,
            cell_offsets,
            sorted_vertices,
        }
    }

    /// Borrows this owned grid as a slice-backed [`FlatSpatialGrid`] without allocation.
    #[must_use]
    #[verified_engine::verified]
    pub fn as_flat_grid(&self) -> FlatSpatialGrid<'_> {
        FlatSpatialGrid {
            origin: self.origin,
            cell_size: self.cell_size,
            grid_dim: self.grid_dim,
            cell_offsets: &self.cell_offsets,
            sorted_vertices: &self.sorted_vertices,
        }
    }

    /// Queries the shortest Euclidean distance from `point` to any vertex in the grid
    /// without dynamic heap allocation.
    #[must_use]
    #[verified_engine::verified]
    pub fn shortest_distance(&self, point: &[f64; 3]) -> f64 {
        self.as_flat_grid().shortest_distance(point)
    }
}

/// Reusable CSR buffer workspace backed by one [`OwnedByteArena`] enabling zero-allocation
/// [`FlatSpatialGrid`] indexing and [`super::cortical_thickness_map_into`] evaluation.
#[derive(Clone, Debug, Default)]
pub struct ThicknessWorkspace {
    arena: OwnedByteArena,
    white_offsets: ArenaRegion<usize>,
    white_vertices: ArenaRegion<[f64; 3]>,
    pial_offsets: ArenaRegion<usize>,
    pial_vertices: ArenaRegion<[f64; 3]>,
    max_vertices: usize,
}

impl ThicknessWorkspace {
    /// Creates an empty unallocated [`ThicknessWorkspace`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arena: OwnedByteArena::new(),
            white_offsets: ArenaRegion::empty(),
            white_vertices: ArenaRegion::empty(),
            pial_offsets: ArenaRegion::empty(),
            pial_vertices: ArenaRegion::empty(),
            max_vertices: 0,
        }
    }

    /// Preallocates CSR offset and vertex buffers for surfaces with up to `max_vertices` vertices.
    #[must_use]
    pub fn with_max_vertices(max_vertices: usize) -> Self {
        let mut ws = Self::new();
        ws.ensure_capacity(max_vertices);
        ws
    }

    /// Grows internal CSR arena capacity if needed to accommodate at least `max_vertices` per surface.
    pub fn ensure_capacity(&mut self, max_vertices: usize) {
        if max_vertices <= self.max_vertices && (max_vertices > 0 || self.arena.byte_capacity() > 0)
        {
            return;
        }
        let offsets_len = FlatSpatialGrid::required_offsets_len(max_vertices);
        let mut layout = ArenaLayout::new();
        self.white_offsets = layout.reserve::<usize>(offsets_len).expect("white_offsets");
        self.white_vertices = layout
            .reserve::<[f64; 3]>(max_vertices)
            .expect("white_vertices");
        self.pial_offsets = layout.reserve::<usize>(offsets_len).expect("pial_offsets");
        self.pial_vertices = layout
            .reserve::<[f64; 3]>(max_vertices)
            .expect("pial_vertices");
        self.arena.ensure_capacity(layout.total_bytes());
        self.max_vertices = max_vertices;
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
        let w_off = self.white_offsets;
        let w_vtx = self.white_vertices;
        let p_off = self.pial_offsets;
        let p_vtx = self.pial_vertices;
        let mut cursor = self.arena.raw_cursor();
        let white_offsets = cursor.region_mut(w_off);
        let white_vertices = cursor.region_mut(w_vtx);
        let pial_offsets = cursor.region_mut(p_off);
        let pial_vertices = cursor.region_mut(p_vtx);
        let white_grid =
            FlatSpatialGrid::from_surface(white_surface, white_offsets, white_vertices)?;
        let pial_grid = FlatSpatialGrid::from_surface(pial_surface, pial_offsets, pial_vertices)?;
        Some((white_grid, pial_grid))
    }
}
