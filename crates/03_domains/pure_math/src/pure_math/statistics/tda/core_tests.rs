use super::*;

#[test]
#[verified_engine::verified]
fn test_point2d_distance() {
    let p1 = Point2D::new(0.0, 0.0);
    let p2 = Point2D::new(3.0, 4.0);

    let d = p1.distance(&p2);
    assert!((d - 5.0).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_point_cloud_empty_and_bounds() {
    assert!(matches!(
        PointCloud::new(vec![]),
        Err(TdaError::EmptyPointCloud)
    ));

    let at_cap: Vec<Point2D> = (0..MAX_TDA_POINTS)
        .map(|i| Point2D::new(i as f64 * 0.1, 0.0))
        .collect();
    assert_eq!(PointCloud::new(at_cap).unwrap().size(), MAX_TDA_POINTS);

    let over_cap: Vec<Point2D> = (0..=MAX_TDA_POINTS)
        .map(|i| Point2D::new(i as f64 * 0.1, 0.0))
        .collect();
    assert!(PointCloud::new(over_cap).is_err());
    assert!(PointCloud::new(vec![Point2D::new(f64::NAN, 0.0)]).is_err());
    assert!(PointCloud::new(vec![Point2D::new(0.0, f64::INFINITY)]).is_err());
}

#[test]
#[verified_engine::verified]
fn test_point_cloud_distance_matrix() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(0.0, 1.0),
    ];
    let cloud = PointCloud::new(points).unwrap();
    let dist = cloud.distance_matrix();

    assert_eq!(dist.len(), 3);
    assert!((dist[0][1] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((dist[0][2] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((dist[1][2] - 2.0_f64.sqrt()).abs() < math_commons::registry::TOLERANCE_FAST);
    assert_eq!(dist[0][1], dist[1][0]);
}

#[test]
#[verified_engine::verified]
fn test_nd_point_cloud_distances_projection_and_limits() {
    let nd = NdPointCloud::new(vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4.0, 6.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 9.0],
    ])
    .unwrap();
    assert_eq!(nd.size(), 3);
    assert_eq!(nd.dimension(), 4);

    let dist = nd.distance_matrix();
    assert!((dist[0][1] - 5.0).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!((dist[0][2] - 5.0).abs() < math_commons::registry::TOLERANCE_FAST);

    let proj = nd.project_2d(0, 3).unwrap();
    assert_eq!(proj.size(), 3);
    assert_eq!(proj.points[2], Point2D::new(1.0, 9.0));
    assert!(nd.project_2d(0, 4).is_err());

    // Jagged row, >32 dimensions, 0 dimensions, and non-finite coordinates fail cleanly
    assert!(NdPointCloud::new(vec![vec![0.0, 1.0], vec![0.0]]).is_err());
    assert!(NdPointCloud::new(vec![vec![0.0; MAX_TDA_DIMENSION + 1]]).is_err());
    assert!(NdPointCloud::new(vec![vec![]]).is_err());
    assert!(NdPointCloud::new(vec![vec![0.0, f64::NAN]]).is_err());
}

#[test]
#[verified_engine::verified]
fn test_simplex_vertex() {
    let s = Simplex::new(vec![0]).unwrap();
    assert_eq!(s.dimension(), 0);
    assert_eq!(s.faces().unwrap().len(), 0);
}

#[test]
#[verified_engine::verified]
fn test_simplex_edge() {
    let s = Simplex::new(vec![0, 1]).unwrap();
    assert_eq!(s.dimension(), 1);
    assert_eq!(s.faces().unwrap().len(), 2);
}

#[test]
#[verified_engine::verified]
fn test_simplex_triangle() {
    let s = Simplex::new(vec![0, 1, 2]).unwrap();
    assert_eq!(s.dimension(), 2);
    assert_eq!(s.faces().unwrap().len(), 3);

    for face in s.faces().unwrap() {
        assert_eq!(face.dimension(), 1);
    }
}

#[test]
#[verified_engine::verified]
fn test_simplex_sorted() {
    let s1 = Simplex::new(vec![2, 0, 1]).unwrap();
    let s2 = Simplex::new(vec![0, 1, 2]).unwrap();
    assert_eq!(s1, s2);
}

#[test]
#[verified_engine::verified]
fn test_simplex_duplicate_vertices() {
    let result = Simplex::new(vec![0, 1, 1]);
    assert!(result.is_err());
}

#[test]
#[verified_engine::verified]
fn test_simplex_diameter() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(0.0, 1.0),
    ];
    let cloud = PointCloud::new(points).unwrap();
    let dist = cloud.distance_matrix();

    let triangle = Simplex::new(vec![0, 1, 2]).unwrap();
    let diameter = triangle.diameter(&dist);
    assert!((diameter - 2.0_f64.sqrt()).abs() < math_commons::registry::TOLERANCE_FAST);
}

// [cite:clinical_trials_statistics]
