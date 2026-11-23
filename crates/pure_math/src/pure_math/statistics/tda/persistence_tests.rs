use super::super::complex::vietoris_rips_complex;
use super::super::core::{NdPointCloud, Point2D, Point3D, PointCloud, PointCloud3D};
use super::super::homology::{betti_number_0, betti_numbers};
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

    let triangle = PointCloud::new(vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(2.0, 0.0),
        Point2D::new(1.0, 1.732),
    ])
    .unwrap();
    let tri_radii: Vec<f64> = (0..30).map(|i| i as f64 * 0.15).collect();
    assert!(
        !compute_persistence(&triangle, &tri_radii)
            .unwrap()
            .is_empty()
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
fn test_compute_persistence_invalid_radius_and_barcode_parity() {
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
    let sort_key = |a: &PersistenceInterval, b: &PersistenceInterval| {
        a.dimension
            .cmp(&b.dimension)
            .then_with(|| a.birth.total_cmp(&b.birth))
            .then_with(|| a.death.total_cmp(&b.death))
    };
    intervals.sort_by(sort_key);
    let mut expected = barcode.intervals.clone();
    expected.sort_by(sort_key);
    assert_eq!(intervals, expected);
}

#[verified_engine::verified]
fn bridged_loops_cloud() -> PointCloud {
    let a = 0.70_f64;
    let b = (1.05_f64 * 1.05 - a * a).sqrt();
    let h = 1.075_f64;
    let dx = (1.25_f64 * 1.25 - h * h).sqrt();
    let w = 0.60 + dx;
    let center_x = a + 0.90 + w;

    PointCloud::new(vec![
        Point2D::new(a, 0.0),
        Point2D::new(0.0, b),
        Point2D::new(-a, 0.0),
        Point2D::new(0.0, -b),
        Point2D::new(center_x - w, 0.0),
        Point2D::new(center_x - 0.60, h),
        Point2D::new(center_x + 0.60, h),
        Point2D::new(center_x + w, 0.0),
        Point2D::new(center_x + 0.60, -h),
        Point2D::new(center_x - 0.60, -h),
    ])
    .unwrap()
}

#[test]
#[verified_engine::verified]
fn test_bridged_loops_exact_pairing_without_lifo_swap() {
    let cloud = bridged_loops_cloud();
    let exact = compute_rips_persistence(&cloud).unwrap();
    let h1: Vec<&RipsInterval> = exact
        .intervals
        .iter()
        .filter(|iv| iv.dimension == 1)
        .collect();

    assert_eq!(h1.len(), 2, "expected two exact H_1 intervals, got {h1:?}");
    assert!((h1[0].birth - 1.05).abs() < 1e-12);
    assert!((h1[0].death - 1.40).abs() < 1e-12);
    assert!(!h1[0].is_essential);

    assert!((h1[1].birth - 1.25).abs() < 1e-12);
    assert!((h1[1].death - 2.15).abs() < 1e-12);
    assert!(!h1[1].is_essential);

    let radii: Vec<f64> = (0..=50).map(|k| k as f64 * 0.05).collect();
    let sampled = compute_persistence(&cloud, &radii).unwrap();
    let mut sampled_h1 = sampled.filter_by_dimension(1);
    sampled_h1.sort_by(|x, y| x.birth.total_cmp(&y.birth));
    assert_eq!(sampled_h1.len(), 2);
    assert!((sampled_h1[0].birth - 1.05).abs() < 1e-12);
    assert!((sampled_h1[0].death - 1.40).abs() < 1e-12);
    assert!((sampled_h1[1].birth - 1.25).abs() < 1e-12);
    assert!((sampled_h1[1].death - 2.15).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_simultaneous_birth_and_death_transitions_detected() {
    let cloud = PointCloud::new(vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(0.8, 0.0),
        Point2D::new(0.8, 0.8),
        Point2D::new(0.0, 0.8),
        Point2D::new(10.0, 0.0),
        Point2D::new(11.3, 0.0),
        Point2D::new(11.3, 1.3),
        Point2D::new(10.0, 1.3),
    ])
    .unwrap();

    let coarse_radii = [0.0, 0.5, 1.0, 1.5, 2.0];
    let barcode = compute_persistence(&cloud, &coarse_radii).unwrap();
    let mut h1 = barcode.filter_by_dimension(1);
    h1.sort_by(|a, b| a.birth.total_cmp(&b.birth));

    assert_eq!(
        h1.len(),
        2,
        "must capture both loops across simultaneous transition: {h1:?}"
    );
    assert_eq!((h1[0].birth, h1[0].death), (1.0, 1.5));
    assert_eq!((h1[1].birth, h1[1].death), (1.5, 2.0));
}

#[test]
#[verified_engine::verified]
fn test_betti_curve_matches_rips_intervals_and_nd_cloud() {
    let cloud = bridged_loops_cloud();
    let exact = compute_rips_persistence(&cloud).unwrap();

    for step in [0.2, 0.95, 1.10, 1.30, 1.45, 2.0, 2.2] {
        let complex = vietoris_rips_complex(&cloud, step).unwrap();
        assert_eq!(
            exact.betti_numbers_at(step),
            betti_numbers(&complex).unwrap()
        );
    }

    let nd = NdPointCloud::new(vec![
        vec![0.0, 0.0, 0.0, 5.0],
        vec![0.0, 1.0, 0.0, 5.0],
        vec![0.0, 1.0, 1.0, 5.0],
        vec![0.0, 0.0, 1.0, 5.0],
    ])
    .unwrap();
    let nd_res = compute_rips_persistence_nd(&nd).unwrap();
    let nd_h1: Vec<_> = nd_res
        .intervals
        .iter()
        .filter(|i| i.dimension == 1)
        .collect();
    assert_eq!(nd_h1.len(), 1);
    assert!((nd_h1[0].birth - 1.0).abs() < 1e-12);
    assert!((nd_h1[0].death - 2.0_f64.sqrt()).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_triangle_cap_sets_truncated_flag() {
    let pts: Vec<Point2D> = (0..6)
        .map(|i| {
            let ang = std::f64::consts::TAU * (i as f64) / 6.0;
            Point2D::new(ang.cos(), ang.sin())
        })
        .collect();
    let dist = PointCloud::new(pts).unwrap().distance_matrix();
    let res = compute_rips_persistence_bounded(&dist, 5).unwrap();
    assert!(res.truncated_triangles);
}
