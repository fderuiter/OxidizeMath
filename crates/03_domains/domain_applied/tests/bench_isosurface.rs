//! Integration tests for bench isosurface.
#[cfg(not(target_arch = "wasm32"))]
use domain_applied::applied::isosurface::{IsosurfaceWorkspace, Mesh, extract_isosurface_into};
use domain_applied::applied::isosurface::{Point3D, VoxelGrid, extract_isosurface};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::allocator::VerifiedAllocator;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::engine::VerifiedEngine;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::metrics::ComplexityMetrics;

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

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

#[cfg(not(target_arch = "wasm32"))]
fn assert_isosurface_bench_invariants(mesh: &Mesh, metrics: &ComplexityMetrics) {
    assert_eq!(mesh.vertices.len(), 21_030);
    assert_eq!(mesh.normals.len(), 21_030);
    assert_eq!(mesh.indices.len(), 126_168);
    assert_eq!(mesh.indices.len() % 3, 0);
    assert!(
        mesh.vertices
            .iter()
            .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
    );
    assert!(
        mesh.normals
            .iter()
            .all(|n| n.x.is_finite() && n.y.is_finite() && n.z.is_finite())
    );
    let vertex_count = mesh.vertices.len();
    assert!(mesh.indices.iter().all(|&idx| idx < vertex_count));
    assert!(metrics.function_calls > 0);
    assert!(metrics.arithmetic_ops > 0);
    assert!(metrics.memory_loads > 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::disallowed_methods)]
#[test]
#[verified_engine::verified]
fn bench_isosurface_performance() {
    let size = 100; // 100^3 = 1M voxels
    let radius = size as f32 / 3.0;
    let grid = make_sphere_grid(size);
    let mut workspace = IsosurfaceWorkspace::with_locked_capacity(size, size);
    let mut mesh = Mesh::with_capacity(32_768, 196_608);

    let start = Instant::now();
    let iterations = 10;
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..iterations {
            extract_isosurface_into(&grid, radius, &mut workspace, &mut mesh)
                .expect("zero-allocation isosurface extraction must succeed");
        }
    });
    let duration = start.elapsed();

    assert_isosurface_bench_invariants(&mesh, &metrics);
    println!("Total time for {} iterations: {:?}", iterations, duration);
    println!(
        "Average time per iteration: {:?}",
        duration / iterations as u32
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn bench_isosurface_aborts_if_hot_loop_allocates() {
    const PROBE_ENV: &str = "OXIDIZE_ISOSURFACE_ALLOC_ABORT_PROBE";
    if std::env::var_os(PROBE_ENV).is_some() {
        let grid = make_sphere_grid(10);
        VerifiedEngine::run_verified(|| {
            let mesh = extract_isosurface(&grid, 3.0).expect("allocating extraction");
            std::hint::black_box(mesh.vertices.len());
        });
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .env(PROBE_ENV, "1")
        .args(["--exact", "bench_isosurface_aborts_if_hot_loop_allocates"])
        .output()
        .expect("spawn isosurface allocation probe");
    assert!(
        !output.status.success(),
        "VerifiedAllocator must abort when allocating extract_isosurface runs inside run_verified"
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
