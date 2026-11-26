use super::*;

#[test]
#[verified_engine::verified]
fn test_point3d_distance_and_finiteness() {
    let p0 = Point3D::new(0.0, 0.0, 0.0);
    let p1 = Point3D::new(2.0, 3.0, 6.0);
    assert!(p0.is_finite());
    assert!(!Point3D::new(f64::NAN, 0.0, 0.0).is_finite());
    assert!((p0.distance(&p1) - 7.0).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_point_cloud_3d_validation_and_distances() {
    assert!(matches!(
        PointCloud3D::new(vec![]),
        Err(TdaError::EmptyPointCloud)
    ));
    assert!(matches!(
        PointCloud3D::new(vec![Point3D::new(f64::INFINITY, 0.0, 0.0)]),
        Err(TdaError::MatrixError { .. })
    ));

    let cloud = PointCloud3D::new(vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(1.0, 0.0, 0.0),
        Point3D::new(0.0, 1.0, 1.0),
    ])
    .unwrap();
    let dist = cloud.distance_matrix();
    assert_eq!(dist.len(), 3);
    assert!((dist[0][1] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((dist[0][2] - 2.0_f64.sqrt()).abs() < math_commons::registry::TOLERANCE_FAST);
    assert_eq!(dist[1][2], dist[2][1]);
}

#[test]
#[verified_engine::verified]
fn test_farthest_point_landmarks_preserves_extremes_and_caps_size() {
    assert!(matches!(
        farthest_point_landmarks(&[], 4),
        Err(TdaError::EmptyPointCloud)
    ));
    let pts = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(0.05, 0.0, 0.0),
        Point3D::new(0.10, 0.0, 0.0),
        Point3D::new(10.0, 0.0, 0.0),
        Point3D::new(0.0, 10.0, 0.0),
        Point3D::new(0.0, 0.0, 10.0),
    ];
    assert!(matches!(
        farthest_point_landmarks(&pts, 0),
        Err(TdaError::InsufficientPoints { .. })
    ));

    let landmarks = farthest_point_landmarks(&pts, 4).unwrap();
    assert_eq!(landmarks.len(), 4);
    assert!(landmarks.contains(&Point3D::new(0.0, 0.0, 0.0)));
    assert!(landmarks.contains(&Point3D::new(10.0, 0.0, 0.0)));
    assert!(landmarks.contains(&Point3D::new(0.0, 10.0, 0.0)));
    assert!(landmarks.contains(&Point3D::new(0.0, 0.0, 10.0)));
}

#[test]
#[verified_engine::verified]
fn test_takens_delay_embedding_3d_coordinates_and_errors() {
    let series = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
    assert!(matches!(
        takens_delay_embedding_3d(&series, 0),
        Err(TdaError::InvalidDimension { dimension: 0 })
    ));
    assert!(matches!(
        takens_delay_embedding_3d(&series[..4], 2),
        Err(TdaError::InsufficientPoints {
            required: 5,
            actual: 4
        })
    ));
    assert!(matches!(
        takens_delay_embedding_3d(&[1.0, f64::NAN, 3.0], 1),
        Err(TdaError::MatrixError { .. })
    ));

    let cloud = takens_delay_embedding_3d(&series, 2).unwrap();
    assert_eq!(cloud.size(), 3);
    assert_eq!(cloud.points[0], Point3D::new(50.0, 30.0, 10.0));
    assert_eq!(cloud.points[1], Point3D::new(60.0, 40.0, 20.0));
    assert_eq!(cloud.points[2], Point3D::new(70.0, 50.0, 30.0));
}
