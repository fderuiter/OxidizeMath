use super::*;
use crate::pure_math::statistics::tda::core::Point2D;
use crate::pure_math::statistics::tda::space3d::Point3D;

#[test]
#[verified_engine::verified]
fn test_simplicial_complex_add_edge() {
    let mut complex = SimplicialComplex::new();
    let edge = Simplex::new(vec![0, 1]).unwrap();
    complex.add_simplex(edge).unwrap();

    assert_eq!(complex.count_simplices(0), 2);
    assert_eq!(complex.count_simplices(1), 1);
}

#[test]
#[verified_engine::verified]
fn test_simplicial_complex_add_triangle() {
    let mut complex = SimplicialComplex::new();
    let triangle = Simplex::new(vec![0, 1, 2]).unwrap();
    complex.add_simplex(triangle).unwrap();

    assert_eq!(complex.count_simplices(0), 3);
    assert_eq!(complex.count_simplices(1), 3);
    assert_eq!(complex.count_simplices(2), 1);
}

#[test]
#[verified_engine::verified]
fn test_simplicial_complex_contains() {
    let mut complex = SimplicialComplex::new();
    let edge = Simplex::new(vec![0, 1]).unwrap();
    complex.add_simplex(edge.clone()).unwrap();

    assert!(complex.contains(&edge));
    assert!(!complex.contains(&Simplex::new(vec![1, 2]).unwrap()));
}

#[test]
#[verified_engine::verified]
fn test_vietoris_rips_three_points_radii() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(0.0, 1.0),
    ];
    let cloud = PointCloud::new(points).unwrap();

    let small = vietoris_rips_complex(&cloud, 0.5).unwrap();
    assert_eq!(small.count_simplices(0), 3);
    assert_eq!(small.count_simplices(1), 0);

    let medium = vietoris_rips_complex(&cloud, 1.0).unwrap();
    assert_eq!(medium.count_simplices(0), 3);
    assert_eq!(medium.count_simplices(1), 2);
    assert_eq!(medium.count_simplices(2), 0);

    let large = vietoris_rips_complex(&cloud, 2.0).unwrap();
    assert_eq!(large.count_simplices(0), 3);
    assert_eq!(large.count_simplices(1), 3);
    assert_eq!(large.count_simplices(2), 1);
}

#[test]
#[verified_engine::verified]
fn test_vietoris_rips_invalid_radius() {
    let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0)];
    let cloud = PointCloud::new(points).unwrap();

    assert!(vietoris_rips_complex(&cloud, -1.0).is_err());
    assert!(vietoris_rips_complex(&cloud, f64::NAN).is_err());
}

#[test]
#[verified_engine::verified]
fn test_vietoris_rips_square_and_3d() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(1.0, 1.0),
        Point2D::new(0.0, 1.0),
    ];
    let cloud = PointCloud::new(points).unwrap();

    let c1 = vietoris_rips_complex(&cloud, 1.0).unwrap();
    assert_eq!(c1.count_simplices(0), 4);
    assert_eq!(c1.count_simplices(1), 4);
    assert_eq!(c1.count_simplices(2), 0);

    let c2 = vietoris_rips_complex(&cloud, 1.5).unwrap();
    assert_eq!(c2.count_simplices(0), 4);
    assert_eq!(c2.count_simplices(1), 6);
    assert_eq!(c2.count_simplices(2), 4);

    let cloud_3d = PointCloud3D::new(vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(1.0, 0.0, 0.0),
        Point3D::new(0.0, 1.0, 0.0),
        Point3D::new(0.0, 0.0, 1.0),
    ])
    .unwrap();
    let tet = vietoris_rips_complex_3d(&cloud_3d, 1.5).unwrap();
    assert_eq!(tet.count_simplices(2), 4);
}

#[test]
#[verified_engine::verified]
fn test_vietoris_rips_from_distances_and_nd_reuse() {
    let dist = vec![
        vec![0.0, 1.0, 2.0],
        vec![1.0, 0.0, 1.0],
        vec![2.0, 1.0, 0.0],
    ];
    let complex = vietoris_rips_from_distances(&dist, 1.0).unwrap();
    assert_eq!(complex.count_simplices(0), 3);
    assert_eq!(complex.count_simplices(1), 2);
    assert_eq!(complex.count_simplices(2), 0);

    let nd = NdPointCloud::new(vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
    ])
    .unwrap();
    let nd_complex = vietoris_rips_complex_nd(&nd, 1.1).unwrap();
    assert_eq!(nd_complex.count_simplices(0), 4);
    assert_eq!(nd_complex.count_simplices(1), 4);
    assert_eq!(nd_complex.count_simplices(2), 0);

    let nd_dist = nd.distance_matrix();
    let (full, truncated) =
        vietoris_rips_from_distance_matrix(&nd_dist, 1.5, MAX_FILTRATION_TRIANGLES).unwrap();
    assert!(!truncated);
    assert_eq!(full.count_simplices(2), 4);

    let (capped, was_truncated) = vietoris_rips_from_distance_matrix(&nd_dist, 1.5, 2).unwrap();
    assert!(was_truncated);
    assert_eq!(capped.count_simplices(2), 2);
}
