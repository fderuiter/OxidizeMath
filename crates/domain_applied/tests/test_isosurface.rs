//! Integration tests for test isosurface.
#[cfg(test)]
mod tests {
    use domain_applied::applied::isosurface::{Mesh, Point3D, VoxelGrid, extract_isosurface};
    use std::collections::HashMap;
    use verified_engine::allocator::VerifiedAllocator;
    use verified_engine::engine::VerifiedEngine;

    #[global_allocator]
    static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

    fn assert_closed_manifold(mesh: &Mesh) {
        assert!(
            !mesh.indices.is_empty(),
            "Expected non-empty triangle indices"
        );
        assert_eq!(mesh.vertices.len(), mesh.normals.len());

        let (triangles, remainder) = mesh.indices.as_chunks::<3>();
        assert!(remainder.is_empty(), "Triangle index count must be 3 * N");

        let mut directed_edges: HashMap<(usize, usize), usize> = HashMap::new();
        for &[a, b, c] in triangles {
            assert!(a < mesh.vertices.len() && b < mesh.vertices.len() && c < mesh.vertices.len());
            assert!(a != b && b != c && c != a, "Degenerate triangle indices");
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *directed_edges.entry((u, v)).or_insert(0) += 1;
            }
        }

        for (&(u, v), &count) in &directed_edges {
            assert_eq!(count, 1, "Non-manifold duplicate half-edge ({u}, {v})");
            assert_eq!(
                directed_edges.get(&(v, u)).copied(),
                Some(1),
                "Unpaired boundary half-edge ({u}, {v}) indicates missed vertex deduplication"
            );
        }
    }

    fn count_sign_changing_lattice_edges(grid: &VoxelGrid, threshold: f32) -> usize {
        let (w, h, d) = (grid.width(), grid.height(), grid.depth());
        let mut count = 0usize;
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let inside = grid.get(x, y, z) < threshold;
                    if x + 1 < w && (grid.get(x + 1, y, z) < threshold) != inside {
                        count += 1;
                    }
                    if y + 1 < h && (grid.get(x, y + 1, z) < threshold) != inside {
                        count += 1;
                    }
                    if z + 1 < d && (grid.get(x, y, z + 1) < threshold) != inside {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    #[test]
    #[verified_engine::verified]
    fn test_sphere_extraction() {
        // Create a 20x20x20 grid
        let size = 20;
        let mut data = vec![0.0; size * size * size];
        let center = size as f32 / 2.0;
        let radius = 6.0;

        // Fill with signed distance field of a sphere (or just distance)
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let dx = x as f32 - center;
                    let dy = y as f32 - center;
                    let dz = z as f32 - center;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    data[z * size * size + y * size + x] = dist;
                }
            }
        }

        let grid = VoxelGrid::builder()
            .dimensions(size, size, size)
            .data(data)
            .voxel_size(Point3D::new(1.0, 1.0, 1.0))
            .origin(Point3D::new(0.0, 0.0, 0.0))
            .build()
            .unwrap();

        // Extract at radius 6.0
        let mesh = extract_isosurface(&grid, radius).unwrap();

        // A sphere should have many triangles
        assert!(mesh.indices.len() / 3 > 100);
        assert_closed_manifold(&mesh);
        assert_eq!(
            mesh.vertices.len(),
            count_sign_changing_lattice_edges(&grid, radius)
        );

        // Check if vertices are approximately at radius distance from center
        for v in &mesh.vertices {
            let dx = v.x - center;
            let dy = v.y - center;
            let dz = v.z - center;
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                (d - radius).abs() < 1.0,
                "Vertex distance {} not close to radius {}",
                d,
                radius
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_small_grid() {
        let grid = VoxelGrid::builder()
            .dimensions(2, 2, 2)
            .data(vec![0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
            .voxel_size(Point3D::new(1.0, 1.0, 1.0))
            .origin(Point3D::new(0.0, 0.0, 0.0))
            .build()
            .unwrap();
        let mesh = extract_isosurface(&grid, 5.0).unwrap();
        assert!(!mesh.indices.is_empty());
        assert_eq!(
            mesh.vertices.len(),
            count_sign_changing_lattice_edges(&grid, 5.0)
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_non_cubic_asymmetric_grids_preserve_closed_manifold() {
        for (width, height, depth) in [(11, 17, 9), (17, 9, 11), (9, 11, 17)] {
            let cx = width as f32 * 0.5;
            let cy = height as f32 * 0.5;
            let cz = depth as f32 * 0.5;
            let mut data = Vec::with_capacity(width * height * depth);

            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        let dx = (x as f32 - cx) / 2.5;
                        let dy = (y as f32 - cy) / 2.5;
                        let dz = (z as f32 - cz) / 2.5;
                        data.push(dx * dx + dy * dy + dz * dz - 1.0);
                    }
                }
            }

            let grid = VoxelGrid::builder()
                .dimensions(width, height, depth)
                .data(data)
                .voxel_size(Point3D::new(0.5, 0.75, 1.25))
                .origin(Point3D::new(-1.0, 2.0, -0.5))
                .build()
                .unwrap();

            let mesh = extract_isosurface(&grid, 0.0).unwrap();
            assert_closed_manifold(&mesh);
            assert_eq!(
                mesh.vertices.len(),
                count_sign_changing_lattice_edges(&grid, 0.0)
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_boundary_touching_surfaces_and_minimal_slices() {
        // Large sphere clipped against all 6 outer grid faces (x=0, x=w-1, y=0, y=h-1, z=0, z=d-1)
        let (width, height, depth) = (8, 6, 7);
        let (cx, cy, cz) = (3.5_f32, 2.5_f32, 3.0_f32);
        let mut data = Vec::with_capacity(width * height * depth);
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dz = z as f32 - cz;
                    data.push((dx * dx + dy * dy + dz * dz).sqrt() - 4.2);
                }
            }
        }

        let grid = VoxelGrid::builder()
            .dimensions(width, height, depth)
            .data(data)
            .voxel_size(Point3D::new(1.0, 1.0, 1.0))
            .origin(Point3D::new(0.0, 0.0, 0.0))
            .build()
            .unwrap();
        let mesh = extract_isosurface(&grid, 0.0).unwrap();
        assert!(!mesh.vertices.is_empty());
        assert_eq!(mesh.vertices.len(), mesh.normals.len());
        assert_eq!(
            mesh.vertices.len(),
            count_sign_changing_lattice_edges(&grid, 0.0)
        );

        // Minimal 2x2x3 grid where y loop runs once per z slice (tests Z-row reset across z > 0)
        let slim_grid = VoxelGrid::builder()
            .dimensions(2, 2, 3)
            .data(vec![
                -1.0, 1.0, 1.0, -1.0, // z = 0
                1.0, -1.0, -1.0, 1.0, // z = 1
                -1.0, 1.0, 1.0, -1.0, // z = 2
            ])
            .voxel_size(Point3D::new(1.0, 1.0, 1.0))
            .origin(Point3D::new(0.0, 0.0, 0.0))
            .build()
            .unwrap();
        let slim_mesh = extract_isosurface(&slim_grid, 0.0).unwrap();
        assert!(!slim_mesh.indices.is_empty());
        assert_eq!(
            slim_mesh.vertices.len(),
            count_sign_changing_lattice_edges(&slim_grid, 0.0)
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_sparse_interior_islands_across_empty_slices() {
        let (width, height, depth) = (18, 16, 30);
        let mut data = Vec::with_capacity(width * height * depth);

        // Two small disjoint spheres separated by empty z slices (z=0..4, z=10..19, z=25..29 empty)
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let d1 = ((x as f32 - 6.0).powi(2)
                        + (y as f32 - 6.0).powi(2)
                        + (z as f32 - 7.0).powi(2))
                    .sqrt()
                        - 2.1;
                    let d2 = ((x as f32 - 11.0).powi(2)
                        + (y as f32 - 9.0).powi(2)
                        + (z as f32 - 22.0).powi(2))
                    .sqrt()
                        - 2.1;
                    data.push(d1.min(d2));
                }
            }
        }

        let grid = VoxelGrid::builder()
            .dimensions(width, height, depth)
            .data(data)
            .voxel_size(Point3D::new(1.0, 1.0, 1.0))
            .origin(Point3D::new(0.0, 0.0, 0.0))
            .build()
            .unwrap();

        let mesh = extract_isosurface(&grid, 0.0).unwrap();
        assert_closed_manifold(&mesh);
        assert_eq!(
            mesh.vertices.len(),
            count_sign_changing_lattice_edges(&grid, 0.0)
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_empty_grid_zero_heap_allocations() {
        let grid = VoxelGrid::builder()
            .dimensions(16, 12, 10)
            .data(vec![10.0; 16 * 12 * 10])
            .voxel_size(Point3D::new(1.0, 1.0, 1.0))
            .origin(Point3D::new(0.0, 0.0, 0.0))
            .build()
            .unwrap();

        let (mesh, _metrics) =
            VerifiedEngine::run_verified(|| extract_isosurface(&grid, 0.0).unwrap());
        assert!(mesh.vertices.is_empty());
        assert!(mesh.normals.is_empty());
        assert!(mesh.indices.is_empty());
    }
}
