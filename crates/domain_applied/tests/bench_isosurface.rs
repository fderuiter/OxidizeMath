#![allow(missing_docs)]
use domain_applied::applied::isosurface::{Point3D, VoxelGrid, extract_isosurface};
use std::time::Instant;

fn make_sphere_grid(size: usize) -> VoxelGrid {
    let mut data = Vec::with_capacity(size * size * size);
    let center = size as f32 / 2.0;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dz = z as f32 - center;
                let val = (dx * dx + dy * dy + dz * dz).sqrt();
                data.push(val);
            }
        }
    }

    VoxelGrid::builder()
        .dimensions(size, size, size)
        .data(data)
        .voxel_size(Point3D::new(1.0, 1.0, 1.0))
        .origin(Point3D::new(0.0, 0.0, 0.0))
        .build()
        .unwrap()
}

#[test]
#[verified_engine::verified]
fn bench_isosurface_performance() {
    let size = 100; // 100^3 = 1M voxels
    let radius = size as f32 / 3.0;
    let grid = make_sphere_grid(size);

    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        let _ = extract_isosurface(&grid, radius).unwrap();
    }
    let duration = start.elapsed();

    println!("Total time for {} iterations: {:?}", iterations, duration);
    println!(
        "Average time per iteration: {:?}",
        duration / iterations as u32
    );
}

#[test]
#[verified_engine::verified]
fn test_mesh_equivalence_across_grid_configurations() {
    // Benchmark configurations with known baseline vertex counts
    let configs = [
        (10, 222, 1320),
        (20, 822, 4920),
        (50, 5262, 31560),
        (100, 21030, 126168),
    ];

    for (size, expected_vertices, expected_indices) in configs {
        let radius = size as f32 / 3.0;
        let grid = make_sphere_grid(size);
        let mesh = extract_isosurface(&grid, radius).expect("extraction failed");

        assert_eq!(
            mesh.vertices.len(),
            expected_vertices,
            "Vertex count mismatch for grid size {}",
            size
        );
        assert_eq!(
            mesh.indices.len(),
            expected_indices,
            "Index count mismatch for grid size {}",
            size
        );
        assert_eq!(
            mesh.normals.len(),
            expected_vertices,
            "Normal count must match vertex count for grid size {}",
            size
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_sparse_memory_scaling_empty_volume() {
    // Empty volume: all voxels far above threshold, no surface intersection
    let size = 100;
    let grid = VoxelGrid::builder()
        .dimensions(size, size, size)
        .data(vec![100.0; size * size * size])
        .voxel_size(Point3D::new(1.0, 1.0, 1.0))
        .origin(Point3D::new(0.0, 0.0, 0.0))
        .build()
        .unwrap();

    let mesh = extract_isosurface(&grid, 0.0).expect("Empty extraction should succeed");
    assert_eq!(mesh.vertices.len(), 0);
    assert_eq!(mesh.normals.len(), 0);
    assert_eq!(mesh.indices.len(), 0);

    // Empty volume: all voxels below threshold
    let mesh_inside = extract_isosurface(&grid, 200.0).expect("Empty extraction should succeed");
    assert_eq!(mesh_inside.vertices.len(), 0);
    assert_eq!(mesh_inside.normals.len(), 0);
    assert_eq!(mesh_inside.indices.len(), 0);
}
