//! Integration tests for security voxelgrid get.
use domain_applied::applied::isosurface::{Point3D, VoxelGrid};

#[test]
#[verified_engine::verified]
fn test_voxelgrid_panic_on_malformed_data() {
    // Attempt to create a VoxelGrid with insufficient data buffer
    // 10x10x10 = 1000 voxels required, but only 1 provided.
    let grid_result = VoxelGrid::builder()
        .dimensions(10, 10, 10)
        .data(vec![0.0; 1]) // MALFORMED
        .voxel_size(Point3D::new(1.0, 1.0, 1.0))
        .origin(Point3D::new(0.0, 0.0, 0.0))
        .build();

    // The builder should reject this
    assert!(grid_result.is_err());
}
