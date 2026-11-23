//! Integration tests verifying zero-allocation Marching Cubes isosurface extraction.
use domain_applied::applied::isosurface::{
    IsosurfaceWorkspace, Mesh, Point3D, VoxelGrid, extract_isosurface, extract_isosurface_into,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_extract_isosurface_into_zero_allocation_hot_path() {
    let dim = 24;
    let total = dim * dim * dim;
    let mut grid = VoxelGrid::builder()
        .dimensions(dim, dim, dim)
        .data(vec![0.0_f32; total])
        .voxel_size(Point3D::new(0.25, 0.25, 0.25))
        .origin(Point3D::new(-3.0, -3.0, -3.0))
        .build()
        .expect("valid voxel grid");
    let mut workspace = IsosurfaceWorkspace::with_locked_capacity(dim, dim);
    let mut mesh = Mesh::with_capacity(16_384, 49_152);

    VerifiedEngine::run_verified(|| {
        for frame in 0..3 {
            let radius = 1.6 + 0.2 * (frame as f32);
            let data = grid.data_mut();
            for z in 0..dim {
                let wz = -3.0 + (z as f32) * 0.25;
                for y in 0..dim {
                    let wy = -3.0 + (y as f32) * 0.25;
                    for x in 0..dim {
                        let wx = -3.0 + (x as f32) * 0.25;
                        let r = (wx * wx + wy * wy + wz * wz).sqrt();
                        data[(z * dim + y) * dim + x] = r - radius;
                    }
                }
            }
            extract_isosurface_into(&grid, 0.0, &mut workspace, &mut mesh)
                .expect("zero-alloc extraction succeeds");
        }
    });

    let reference = extract_isosurface(&grid, 0.0).expect("reference extraction succeeds");
    assert!(!mesh.indices.is_empty());
    assert_eq!(mesh.vertices.len(), reference.vertices.len());
    assert_eq!(mesh.indices.len(), reference.indices.len());
    assert!(
        mesh.normals
            .iter()
            .all(|n| n.x.is_finite() && n.y.is_finite() && n.z.is_finite())
    );
}
