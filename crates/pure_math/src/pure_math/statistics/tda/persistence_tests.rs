use super::super::complex::vietoris_rips_complex;
use super::super::core::{Point2D, Point3D, PointCloud3D};
use super::super::workspace::TdaFiltrationWorkspace;
use super::*;

#[test]
#[verified_engine::verified]
fn test_persistence_interval_validation_and_significance() {
    let interval = PersistenceInterval::new(0.5, 2.0, 1).unwrap();
    assert_eq!(interval.birth, 0.5);
    assert_eq!(interval.death, 2.0);
    assert_eq!(interval.dimension, 1);
    assert!((interval.persistence() - 1.5).abs() < math_commons::registry::TOLERANCE_FAST);
    assert!(interval.is_significant(1.0));
    assert!(!interval.is_significant(2.0));

    assert!(PersistenceInterval::new(2.0, 1.0, 0).is_err());
    assert!(PersistenceInterval::new(-1.0, 1.0, 0).is_err());
}

#[test]
#[verified_engine::verified]
fn test_barcode_filtering_and_most_persistent() {
    let mut barcode = PersistenceBarcode::new();
    barcode.add_interval(PersistenceInterval::new(0.0, 0.5, 0).unwrap());
    barcode.add_interval(PersistenceInterval::new(0.5, 3.0, 1).unwrap());
    barcode.add_interval(PersistenceInterval::new(1.0, 2.0, 1).unwrap());

    assert_eq!(barcode.filter_by_dimension(0).len(), 1);
    assert_eq!(barcode.filter_by_dimension(1).len(), 2);
    assert_eq!(barcode.filter_by_persistence(1.2).len(), 1);
    assert!(
        (barcode.most_persistent(1).unwrap().persistence() - 2.5).abs()
            < math_commons::registry::TOLERANCE_FAST
    );
}

#[test]
#[verified_engine::verified]
fn test_compute_persistence_line_triangle_and_clusters() {
    let line = PointCloud::new(vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(2.0, 0.0),
    ])
    .unwrap();
    let radii: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
    let line_barcode = compute_persistence(&line, &radii).unwrap();
    assert!(!line_barcode.is_empty());
    assert!(
        line_barcode
            .filter_by_dimension(1)
            .iter()
            .all(|h| !h.is_significant(0.5))
    );

    let clusters = PointCloud::new(vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(10.0, 0.0),
        Point2D::new(11.0, 0.0),
    ])
    .unwrap();
    let cluster_radii: Vec<f64> = (0..50).map(|i| i as f64 * 0.3).collect();
    let cluster_barcode = compute_persistence(&clusters, &cluster_radii).unwrap();
    assert!(
        cluster_barcode
            .filter_by_dimension(0)
            .iter()
            .any(|c| c.is_significant(1.0))
    );
}

#[test]
#[verified_engine::verified]
fn test_compute_persistence_invalid_radius_and_parity() {
    let cloud = PointCloud::new(vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(0.0, 1.0),
        Point2D::new(1.0, 1.0),
    ])
    .unwrap();
    assert!(matches!(
        compute_persistence(&cloud, &[]),
        Err(TdaError::InvalidRadius { .. })
    ));
    assert!(matches!(
        compute_persistence(&cloud, &[-1.0]),
        Err(TdaError::InvalidRadius { .. })
    ));
    assert!(matches!(
        compute_persistence(&cloud, &[f64::NAN]),
        Err(TdaError::InvalidRadius { .. })
    ));

    let radii = [0.0, 0.5, 1.0, 1.5, 2.0];
    let barcode = compute_persistence(&cloud, &radii).unwrap();
    let b0_seq: Vec<usize> = radii
        .iter()
        .map(|&r| betti_number_0(&vietoris_rips_complex(&cloud, r).unwrap()).unwrap())
        .collect();
    assert_eq!(b0_seq, vec![4, 4, 1, 1, 1]);
    assert_eq!(barcode.intervals.len(), 5);
}

#[test]
#[verified_engine::verified]
fn test_compute_persistence_from_distances_3d_and_workspace_parity() {
    let pts = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(2.0, 0.0, 0.0),
        Point3D::new(2.0, 2.0, 0.0),
        Point3D::new(0.0, 2.0, 0.0),
        Point3D::new(0.0, 0.0, 6.0),
        Point3D::new(2.0, 0.0, 6.0),
        Point3D::new(2.0, 2.0, 6.0),
        Point3D::new(0.0, 2.0, 6.0),
    ];
    let cloud = PointCloud3D::new(pts.clone()).unwrap();
    let dist = cloud.distance_matrix();
    let radii: Vec<f64> = (0..24).map(|i| i as f64 * 0.4).collect();
    let barcode = compute_persistence_from_distances(&dist, &radii).unwrap();
    let dim1 = barcode.filter_by_dimension(1);
    // 2 square holes born at r=2.0 (dying at r=3.2) + 3 vertical prism side-holes born at r=6.0 (dying at r=6.4).
    assert_eq!(dim1.len(), 5);
    assert_eq!(
        dim1.iter()
            .filter(|iv| (iv.birth - 2.0).abs() < 1e-9)
            .count(),
        2
    );
    assert_eq!(
        dim1.iter()
            .filter(|iv| (iv.birth - 6.0).abs() < 1e-9)
            .count(),
        3
    );

    let mut ws = TdaFiltrationWorkspace::with_capacity(64);
    let mut intervals = Vec::with_capacity(64);
    ws.load_points_3d(&pts).unwrap();
    ws.compute_persistence_into(pts.len(), &radii, &mut intervals)
        .unwrap();
    assert_eq!(intervals, barcode.intervals);
}
