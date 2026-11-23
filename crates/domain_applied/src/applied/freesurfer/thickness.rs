use super::surface::Surface;

const MAX_GRID_CELL_COORD: f64 = 1.0e12;

#[derive(Clone, Copy, Debug, Default)]
struct CellBucket {
    key: [i64; 3],
    start: usize,
    len: usize,
}

#[inline]
fn hash_cell(key: [i64; 3]) -> usize {
    let mut h = (key[0] as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (key[1] as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= (key[2] as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (h ^ (h >> 31)) as usize
}

#[inline]
fn cell_coords(point: &[f64; 3], cell_size: f64) -> [i64; 3] {
    [
        (point[0] / cell_size).floor() as i64,
        (point[1] / cell_size).floor() as i64,
        (point[2] / cell_size).floor() as i64,
    ]
}

#[inline]
fn point_within_grid_range(point: &[f64; 3], cell_size: f64) -> bool {
    point
        .iter()
        .all(|c| c.is_finite() && (c / cell_size).abs() <= MAX_GRID_CELL_COORD)
}

fn compute_cell_size(vertices: &[[f64; 3]]) -> f64 {
    let mut min_pt = [f64::INFINITY; 3];
    let mut max_pt = [f64::NEG_INFINITY; 3];
    for v in vertices {
        for d in 0..3 {
            min_pt[d] = min_pt[d].min(v[d]);
            max_pt[d] = max_pt[d].max(v[d]);
        }
    }
    let max_extent = (max_pt[0] - min_pt[0])
        .max(max_pt[1] - min_pt[1])
        .max(max_pt[2] - min_pt[2]);
    let m = (vertices.len() as f64).cbrt().ceil().max(1.0);
    if max_extent > 0.0 && max_extent.is_finite() {
        (max_extent / m).max(1e-6)
    } else {
        1.0
    }
}

fn find_or_insert_bucket(buckets: &mut [CellBucket], mask: usize, key: [i64; 3]) -> usize {
    let mut slot = hash_cell(key) & mask;
    loop {
        let entry = &mut buckets[slot];
        if entry.len == 0 {
            entry.key = key;
            entry.len = 1;
            return slot;
        }
        if entry.key == key {
            entry.len += 1;
            return slot;
        }
        slot = (slot + 1) & mask;
    }
}

#[inline]
fn update_min_distance(point: &[f64; 3], verts: &[[f64; 3]], mut min_distance: f64) -> f64 {
    for v in verts {
        let dx = point[0] - v[0];
        let dy = point[1] - v[1];
        let dz = point[2] - v[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        min_distance = min_distance.min(dist);
    }
    min_distance
}

/// Reusable uniform three-dimensional voxel grid backed by a flat CSR vertex buffer
/// and deterministic open-addressed cell lookup table for zero-allocation distance queries.
///
/// # Example
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::freesurfer::{SpatialGrid, Surface};
///
/// let surface = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
/// let grid = SpatialGrid::from_surface(&surface);
/// assert_eq!(grid.shortest_distance(&[0.0, 3.0, 4.0]), 5.0);
/// ```
#[derive(Clone, Debug)]
pub struct SpatialGrid {
    cell_size: f64,
    min_cell: [i64; 3],
    max_cell: [i64; 3],
    buckets: Vec<CellBucket>,
    mask: usize,
    vertices: Vec<[f64; 3]>,
    linear_fallback: bool,
}

impl SpatialGrid {
    /// Constructs a reusable flat CSR spatial uniform grid partitioning `surface` vertices.
    pub fn from_surface(surface: &Surface) -> Self {
        if surface.vertices.is_empty() {
            return Self::unindexed(1.0, Vec::new(), false);
        }
        let cell_size = compute_cell_size(&surface.vertices);
        if !surface
            .vertices
            .iter()
            .all(|v| point_within_grid_range(v, cell_size))
        {
            return Self::unindexed(cell_size, surface.vertices.clone(), true);
        }
        Self::build_csr(&surface.vertices, cell_size)
    }

    fn unindexed(cell_size: f64, vertices: Vec<[f64; 3]>, linear_fallback: bool) -> Self {
        Self {
            cell_size,
            min_cell: [0; 3],
            max_cell: [0; 3],
            buckets: Vec::new(),
            mask: 0,
            vertices,
            linear_fallback,
        }
    }

    fn build_csr(src_vertices: &[[f64; 3]], cell_size: f64) -> Self {
        let n = src_vertices.len();
        let cap = n.saturating_mul(2).next_power_of_two().max(8);
        let mask = cap - 1;
        let mut buckets = vec![CellBucket::default(); cap];
        let mut vertex_slots = Vec::with_capacity(n);
        let mut min_cell = [i64::MAX; 3];
        let mut max_cell = [i64::MIN; 3];

        for v in src_vertices {
            let key = cell_coords(v, cell_size);
            for d in 0..3 {
                min_cell[d] = min_cell[d].min(key[d]);
                max_cell[d] = max_cell[d].max(key[d]);
            }
            vertex_slots.push(find_or_insert_bucket(&mut buckets, mask, key));
        }

        let mut offset = 0;
        for bucket in &mut buckets {
            if bucket.len > 0 {
                bucket.start = offset;
                offset += bucket.len;
                bucket.len = 0;
            }
        }

        let mut vertices = vec![[0.0; 3]; n];
        for (v, slot) in src_vertices.iter().zip(vertex_slots) {
            let bucket = &mut buckets[slot];
            let write_idx = bucket.start + bucket.len;
            vertices[write_idx] = *v;
            bucket.len += 1;
        }

        Self {
            cell_size,
            min_cell,
            max_cell,
            buckets,
            mask,
            vertices,
            linear_fallback: false,
        }
    }

    #[inline]
    fn cell_vertices(&self, key: [i64; 3]) -> &[[f64; 3]] {
        if self.buckets.is_empty() {
            return &[];
        }
        let mut slot = hash_cell(key) & self.mask;
        loop {
            let bucket = &self.buckets[slot];
            if bucket.len == 0 {
                return &[];
            }
            if bucket.key == key {
                return &self.vertices[bucket.start..bucket.start + bucket.len];
            }
            slot = (slot + 1) & self.mask;
        }
    }

    #[inline]
    fn initial_radius(&self, center: [i64; 3]) -> i64 {
        let rx = (self.min_cell[0] - center[0]).max(center[0] - self.max_cell[0]);
        let ry = (self.min_cell[1] - center[1]).max(center[1] - self.max_cell[1]);
        let rz = (self.min_cell[2] - center[2]).max(center[2] - self.max_cell[2]);
        rx.max(ry).max(rz).max(0)
    }

    #[inline]
    fn is_terminal_shell(
        &self,
        point: &[f64; 3],
        center: [i64; 3],
        radius: i64,
        min_distance: f64,
    ) -> bool {
        let s = self.cell_size;
        let dist_to_boundary = (point[0] - (center[0] - radius) as f64 * s)
            .min((center[0] + radius + 1) as f64 * s - point[0])
            .min(point[1] - (center[1] - radius) as f64 * s)
            .min((center[1] + radius + 1) as f64 * s - point[1])
            .min(point[2] - (center[2] - radius) as f64 * s)
            .min((center[2] + radius + 1) as f64 * s - point[2]);
        let covered_all = (center[0] - radius <= self.min_cell[0])
            && (center[0] + radius >= self.max_cell[0])
            && (center[1] - radius <= self.min_cell[1])
            && (center[1] + radius >= self.max_cell[1])
            && (center[2] - radius <= self.min_cell[2])
            && (center[2] + radius >= self.max_cell[2]);
        min_distance <= dist_to_boundary || covered_all
    }

    #[inline]
    fn probe_cell(&self, point: &[f64; 3], key: [i64; 3], min_distance: f64) -> f64 {
        update_min_distance(point, self.cell_vertices(key), min_distance)
    }

    fn search_shell(&self, point: &[f64; 3], center: [i64; 3], radius: i64, mut d: f64) -> f64 {
        let x_lo = (center[0] - radius).max(self.min_cell[0]);
        let x_hi = (center[0] + radius).min(self.max_cell[0]);
        let y_lo = (center[1] - radius).max(self.min_cell[1]);
        let y_hi = (center[1] + radius).min(self.max_cell[1]);
        let z_lo = (center[2] - radius).max(self.min_cell[2]);
        let z_hi = (center[2] + radius).min(self.max_cell[2]);
        let (z_neg, z_pos) = (center[2] - radius, center[2] + radius);

        for kx in x_lo..=x_hi {
            let x_edge = (kx - center[0]).abs() == radius;
            for ky in y_lo..=y_hi {
                if x_edge || (ky - center[1]).abs() == radius {
                    for kz in z_lo..=z_hi {
                        d = self.probe_cell(point, [kx, ky, kz], d);
                    }
                } else {
                    if (z_lo..=z_hi).contains(&z_neg) {
                        d = self.probe_cell(point, [kx, ky, z_neg], d);
                    }
                    if (z_lo..=z_hi).contains(&z_pos) {
                        d = self.probe_cell(point, [kx, ky, z_pos], d);
                    }
                }
            }
        }
        d
    }

    /// Queries the shortest Euclidean distance from `point` to any vertex in the grid
    /// without dynamic heap allocation.
    #[verified_engine::verified]
    pub fn shortest_distance(&self, point: &[f64; 3]) -> f64 {
        if self.vertices.is_empty() {
            return f64::INFINITY;
        }
        if point[0].is_nan() || point[1].is_nan() || point[2].is_nan() {
            return f64::NAN;
        }
        if self.linear_fallback || !point_within_grid_range(point, self.cell_size) {
            return update_min_distance(point, &self.vertices, f64::INFINITY);
        }

        let center = [
            (point[0] / self.cell_size).floor() as i64,
            (point[1] / self.cell_size).floor() as i64,
            (point[2] / self.cell_size).floor() as i64,
        ];
        let mut min_distance = f64::INFINITY;
        let mut radius = self.initial_radius(center);

        loop {
            min_distance = self.search_shell(point, center, radius, min_distance);
            if self.is_terminal_shell(point, center, radius, min_distance) {
                break;
            }
            radius += 1;
        }
        min_distance
    }
}

/// Calculates the shortest Euclidean distance from a point to a surface without heap allocation.
#[verified_engine::verified]
fn point_to_surface_distance(point: &[f64; 3], surface: &Surface) -> f64 {
    if surface.vertices.is_empty() {
        return f64::INFINITY;
    }
    if point[0].is_nan() || point[1].is_nan() || point[2].is_nan() {
        return f64::NAN;
    }
    let mut min_distance = f64::INFINITY;
    for v in &surface.vertices {
        let dx = point[0] - v[0];
        let dy = point[1] - v[1];
        let dz = point[2] - v[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        min_distance = min_distance.min(dist);
    }
    min_distance
}

/// Calculates cortical thickness at a single vertex pair without dynamic memory allocation.
///
/// For repeated or whole-mesh queries across $V$ vertices, prefer pre-indexing both surfaces once
/// with [`SpatialGrid::from_surface`] and calling [`cortical_thickness_with_grid`],
/// [`cortical_thickness_into`], [`cortical_thickness_mesh`], or [`cortical_thickness_map`].
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

/// Calculates cortical thickness at a vertex pair using pre-built [`SpatialGrid`] indices
/// with zero post-initialization heap allocations.
#[verified_engine::verified]
pub fn cortical_thickness_with_grid(
    v_w: &[f64; 3],
    v_p: &[f64; 3],
    white_grid: &SpatialGrid,
    pial_grid: &SpatialGrid,
) -> f64 {
    let dist_w_to_p = pial_grid.shortest_distance(v_w);
    let dist_p_to_w = white_grid.shortest_distance(v_p);
    0.5 * (dist_w_to_p + dist_p_to_w)
}

/// Computes vertex-wise cortical thickness across `white_surface` and `pial_surface` into
/// caller-allocated `out` using pre-built [`SpatialGrid`] indices with zero heap allocations.
///
/// Evaluates `count = white_surface.vertices.len().min(pial_surface.vertices.len()).min(out.len())`
/// corresponding vertex pairs and zeros any trailing elements in `out[count..]`.
#[verified_engine::verified]
pub fn cortical_thickness_into(
    white_surface: &Surface,
    pial_surface: &Surface,
    white_grid: &SpatialGrid,
    pial_grid: &SpatialGrid,
    out: &mut [f64],
) {
    let count = white_surface
        .vertices
        .len()
        .min(pial_surface.vertices.len())
        .min(out.len());
    for ((dst, v_w), v_p) in out[..count]
        .iter_mut()
        .zip(&white_surface.vertices[..count])
        .zip(&pial_surface.vertices[..count])
    {
        *dst = cortical_thickness_with_grid(v_w, v_p, white_grid, pial_grid);
    }
    out[count..].fill(0.0);
}

/// Computes vertex-wise cortical thickness across an entire cortical mesh pair by constructing
/// one reusable [`SpatialGrid`] per surface.
pub fn cortical_thickness_mesh(white_surface: &Surface, pial_surface: &Surface) -> Vec<f64> {
    let count = white_surface
        .vertices
        .len()
        .min(pial_surface.vertices.len());
    if count == 0 {
        return Vec::new();
    }
    let (w_grid, p_grid) = (
        SpatialGrid::from_surface(white_surface),
        SpatialGrid::from_surface(pial_surface),
    );
    let mut out = vec![0.0; count];
    cortical_thickness_into(white_surface, pial_surface, &w_grid, &p_grid, &mut out);
    out
}

/// Computes symmetric cortical thickness across all corresponding `(white, pial)` vertex pairs,
/// constructing a uniform [`SpatialGrid`] once per surface mesh.
#[must_use]
#[verified_engine::verified]
pub fn cortical_thickness_map(white_surface: &Surface, pial_surface: &Surface) -> Vec<f64> {
    cortical_thickness_mesh(white_surface, pial_surface)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dist(pt: &[f64; 3], surface: &Surface) -> f64 {
        surface
            .vertices
            .iter()
            .map(|v| {
                let (dx, dy, dz) = (pt[0] - v[0], pt[1] - v[1], pt[2] - v[2]);
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn test_empty_and_zero_extent_surfaces() {
        let empty = Surface::from_vertices(vec![]);
        let empty_grid = SpatialGrid::from_surface(&empty);
        let q0 = [1.0, 2.0, 3.0];
        assert_eq!(point_to_surface_distance(&q0, &empty), f64::INFINITY);
        assert_eq!(empty_grid.shortest_distance(&q0), f64::INFINITY);

        let single = Surface::from_vertices(vec![q0]);
        let coincident = Surface::from_vertices(vec![q0; 3]);
        let g1 = SpatialGrid::from_surface(&single);
        let g2 = SpatialGrid::from_surface(&coincident);
        let q1 = [4.0, 6.0, 3.0];
        assert_eq!((g1.cell_size, g2.cell_size), (1.0, 1.0));
        assert_eq!(point_to_surface_distance(&q1, &single), 5.0);
        assert_eq!(g1.shortest_distance(&q1), 5.0);
        assert_eq!(g2.shortest_distance(&q1), 5.0);
    }

    #[test]
    fn test_nan_and_infinite_queries() {
        let surface = Surface::from_vertices(vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]);
        let grid = SpatialGrid::from_surface(&surface);
        for d in 0..3 {
            let mut nan_pt = [0.0; 3];
            nan_pt[d] = f64::NAN;
            assert!(point_to_surface_distance(&nan_pt, &surface).is_nan());
            assert!(grid.shortest_distance(&nan_pt).is_nan());
        }
        let inf_pt = [f64::INFINITY, 0.0, 0.0];
        assert_eq!(point_to_surface_distance(&inf_pt, &surface), f64::INFINITY);
        assert_eq!(grid.shortest_distance(&inf_pt), f64::INFINITY);
    }

    #[test]
    fn test_exact_distance_equivalence_random_and_sparse() {
        let vertices: Vec<[f64; 3]> = (0..100)
            .map(|i| {
                [
                    ((i * 17) % 100) as f64 * 0.1,
                    ((i * 31) % 100) as f64 * 0.1,
                    ((i * 47) % 100) as f64 * 0.1,
                ]
            })
            .collect();
        let surface = Surface::from_vertices(vertices);
        let grid = SpatialGrid::from_surface(&surface);
        for j in 0..50 {
            let q = [
                ((j * 23) % 120) as f64 * 0.1 - 1.0,
                ((j * 41) % 120) as f64 * 0.1 - 1.0,
                ((j * 59) % 120) as f64 * 0.1 - 1.0,
            ];
            let expected = naive_dist(&q, &surface);
            assert_eq!(expected, point_to_surface_distance(&q, &surface));
            assert_eq!(expected, grid.shortest_distance(&q));
        }

        let sparse = Surface::from_vertices(vec![[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]]);
        let sparse_grid = SpatialGrid::from_surface(&sparse);
        for q in [[0.0, 0.0, 0.0], [500.0, -300.0, 200.0]] {
            assert_eq!(naive_dist(&q, &sparse), sparse_grid.shortest_distance(&q));
        }
        assert_eq!(
            cortical_thickness_map(&surface, &sparse),
            cortical_thickness_mesh(&surface, &sparse)
        );
    }
}
