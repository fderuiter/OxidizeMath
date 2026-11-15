use super::surface::Surface;
use std::collections::HashMap;

/// Uniform three-dimensional voxel grid over surface vertices to accelerate distance queries.
#[derive(Debug)]
pub(crate) struct SpatialGrid {
    cell_size: f64,
    min_cell: (i64, i64, i64),
    max_cell: (i64, i64, i64),
    cells: HashMap<(i64, i64, i64), Vec<[f64; 3]>>,
}

impl SpatialGrid {
    /// Constructs a spatial uniform grid partitioning the surface vertices.
    pub(crate) fn from_surface(surface: &Surface) -> Self {
        if surface.vertices.is_empty() {
            return Self {
                cell_size: 1.0,
                min_cell: (0, 0, 0),
                max_cell: (0, 0, 0),
                cells: HashMap::new(),
            };
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        for v in &surface.vertices {
            min_x = min_x.min(v[0]);
            max_x = max_x.max(v[0]);
            min_y = min_y.min(v[1]);
            max_y = max_y.max(v[1]);
            min_z = min_z.min(v[2]);
            max_z = max_z.max(v[2]);
        }

        let extent_x = max_x - min_x;
        let extent_y = max_y - min_y;
        let extent_z = max_z - min_z;
        let max_extent = extent_x.max(extent_y).max(extent_z);

        let n = surface.vertices.len();
        // Determine grid cell size based on surface bounding box and vertex count
        let m = (n as f64).cbrt().ceil().max(1.0);
        let cell_size = if max_extent > 0.0 {
            (max_extent / m).max(1e-6)
        } else {
            1.0
        };

        let mut cells: HashMap<(i64, i64, i64), Vec<[f64; 3]>> = HashMap::with_capacity(n);
        let mut min_c = (i64::MAX, i64::MAX, i64::MAX);
        let mut max_c = (i64::MIN, i64::MIN, i64::MIN);

        for &v in &surface.vertices {
            let cx = (v[0] / cell_size).floor() as i64;
            let cy = (v[1] / cell_size).floor() as i64;
            let cz = (v[2] / cell_size).floor() as i64;

            min_c.0 = min_c.0.min(cx);
            min_c.1 = min_c.1.min(cy);
            min_c.2 = min_c.2.min(cz);
            max_c.0 = max_c.0.max(cx);
            max_c.1 = max_c.1.max(cy);
            max_c.2 = max_c.2.max(cz);

            cells.entry((cx, cy, cz)).or_default().push(v);
        }

        Self {
            cell_size,
            min_cell: min_c,
            max_cell: max_c,
            cells,
        }
    }

    /// Queries the shortest Euclidean distance from a query point to any vertex in the grid.
    pub(crate) fn shortest_distance(&self, point: &[f64; 3]) -> f64 {
        if self.cells.is_empty() {
            return f64::INFINITY;
        }

        if point[0].is_nan() || point[1].is_nan() || point[2].is_nan() {
            return f64::NAN;
        }

        let cx = (point[0] / self.cell_size).floor() as i64;
        let cy = (point[1] / self.cell_size).floor() as i64;
        let cz = (point[2] / self.cell_size).floor() as i64;

        let mut min_distance = f64::INFINITY;
        let mut radius: i64 = 0;

        loop {
            // Search candidate vertices in concentric shells (adjacent cells before outer boundaries)
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        if radius > 0 && dx.abs() < radius && dy.abs() < radius && dz.abs() < radius
                        {
                            // Inner cells were already searched in previous rounds
                            continue;
                        }
                        let cell_key = (cx + dx, cy + dy, cz + dz);
                        if let Some(verts) = self.cells.get(&cell_key) {
                            for v in verts {
                                let diff_x = point[0] - v[0];
                                let diff_y = point[1] - v[1];
                                let diff_z = point[2] - v[2];
                                let dist =
                                    (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z).sqrt();
                                if dist < min_distance {
                                    min_distance = dist;
                                }
                            }
                        }
                    }
                }
            }

            // Minimum distance from the query point to the outer boundaries of the searched grid box
            let min_x_searched = (cx - radius) as f64 * self.cell_size;
            let max_x_searched = (cx + radius + 1) as f64 * self.cell_size;
            let min_y_searched = (cy - radius) as f64 * self.cell_size;
            let max_y_searched = (cy + radius + 1) as f64 * self.cell_size;
            let min_z_searched = (cz - radius) as f64 * self.cell_size;
            let max_z_searched = (cz + radius + 1) as f64 * self.cell_size;

            let dist_to_boundary = (point[0] - min_x_searched)
                .min(max_x_searched - point[0])
                .min(point[1] - min_y_searched)
                .min(max_y_searched - point[1])
                .min(point[2] - min_z_searched)
                .min(max_z_searched - point[2]);

            // Check if all populated cells in the grid have been completely covered
            let covered_all = (cx - radius <= self.min_cell.0)
                && (cx + radius >= self.max_cell.0)
                && (cy - radius <= self.min_cell.1)
                && (cy + radius >= self.max_cell.1)
                && (cz - radius <= self.min_cell.2)
                && (cz + radius >= self.max_cell.2);

            if min_distance <= dist_to_boundary || covered_all {
                break;
            }

            radius += 1;
        }

        min_distance
    }
}

/// Calculates the shortest Euclidean distance from a point to a surface.
#[verified_engine::verified]
fn point_to_surface_distance(point: &[f64; 3], surface: &Surface) -> f64 {
    let grid = SpatialGrid::from_surface(surface);
    grid.shortest_distance(point)
}

/// Calculates cortical thickness at a specific vertex.
///
/// # Arguments
/// * `v_w` - A vertex on the white matter surface.
/// * `v_p` - The corresponding vertex on the pial surface.
/// * `white_surface` - The entire white matter surface.
/// * `pial_surface` - The entire pial surface.
///
/// # Returns
/// The cortical thickness at that location.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_point_to_surface_distance(point: &[f64; 3], surface: &Surface) -> f64 {
        surface
            .vertices
            .iter()
            .map(|v| {
                let dx = point[0] - v[0];
                let dy = point[1] - v[1];
                let dz = point[2] - v[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn test_empty_surface() {
        let surface = Surface { vertices: vec![] };
        let point = [1.0, 2.0, 3.0];
        assert_eq!(point_to_surface_distance(&point, &surface), f64::INFINITY);
    }

    #[test]
    fn test_single_vertex() {
        let surface = Surface {
            vertices: vec![[1.0, 2.0, 3.0]],
        };
        let point = [4.0, 6.0, 3.0];
        // sqrt((4-1)^2 + (6-2)^2 + (3-3)^2) = sqrt(9 + 16) = 5.0
        assert_eq!(point_to_surface_distance(&point, &surface), 5.0);
    }

    #[test]
    fn test_exact_distance_equivalence_random() {
        // Deterministic pseudo-random sequence
        let mut vertices = Vec::new();
        for i in 0..100 {
            let x = ((i * 17) % 100) as f64 * 0.1;
            let y = ((i * 31) % 100) as f64 * 0.1;
            let z = ((i * 47) % 100) as f64 * 0.1;
            vertices.push([x, y, z]);
        }
        let surface = Surface { vertices };

        for j in 0..50 {
            let px = ((j * 23) % 120) as f64 * 0.1 - 1.0;
            let py = ((j * 41) % 120) as f64 * 0.1 - 1.0;
            let pz = ((j * 59) % 120) as f64 * 0.1 - 1.0;
            let query = [px, py, pz];

            let expected = naive_point_to_surface_distance(&query, &surface);
            let actual = point_to_surface_distance(&query, &surface);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_sparse_vertices_expansion() {
        // Query point far from any vertex to ensure radius expands until candidate is found
        let surface = Surface {
            vertices: vec![[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]],
        };
        let query = [0.0, 0.0, 0.0];
        let expected = naive_point_to_surface_distance(&query, &surface);
        let actual = point_to_surface_distance(&query, &surface);
        assert_eq!(expected, actual);
    }
}
