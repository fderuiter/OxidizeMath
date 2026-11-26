//! Integration tests for security isosurface overflow.
use domain_applied::applied::isosurface::{Point3D, VoxelGrid};

#[test]
#[verified_engine::verified]
fn test_security_buffer_overflow_prevention() {
    // Attempt to create a VoxelGrid with insufficient data buffer
    // Dimensions 10x10x10 require 1000 elements.
    // We only provide 10.
    let grid_result = VoxelGrid::builder()
        .dimensions(10, 10, 10)
        .data(vec![0.0; 10]) // MALFORMED: Too small!
        .voxel_size(Point3D::new(1.0, 1.0, 1.0))
        .origin(Point3D::new(0.0, 0.0, 0.0))
        .build();

    // With the builder, it should return an Err before we can even call extract_isosurface.
    assert!(
        grid_result.is_err(),
        "Should return error for insufficient buffer size"
    );
}
