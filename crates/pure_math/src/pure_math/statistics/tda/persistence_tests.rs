use super::super::complex::vietoris_rips_complex;
use super::super::core::{NdPointCloud, Point2D, PointCloud};
use super::super::homology::{betti_number_0, betti_numbers};
use super::*;

#[test]
#[verified_engine::verified]
fn test_persistence_interval() {
    let interval = PersistenceInterval::new(0.5, 2.0, 1).unwrap();
    assert_eq!(interval.birth, 0.5);
    assert_eq!(interval.death, 2.0);
    assert_eq!(interval.dimension, 1);
    assert!((interval.persistence() - 1.5).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_persistence_interval_invalid() {
    assert!(PersistenceInterval::new(2.0, 1.0, 0).is_err());
    assert!(PersistenceInterval::new(-1.0, 1.0, 0).is_err());
}

#[test]
#[verified_engine::verified]
fn test_persistence_interval_is_significant() {
    let interval = PersistenceInterval::new(0.5, 2.0, 1).unwrap();
    assert!(interval.is_significant(1.0));
    assert!(!interval.is_significant(2.0));
}

#[test]
#[verified_engine::verified]
fn test_barcode_filter_by_dimension() {
    let mut barcode = PersistenceBarcode::new();
    barcode.add_interval(PersistenceInterval::new(0.0, 1.0, 0).unwrap());
    barcode.add_interval(PersistenceInterval::new(0.5, 2.0, 1).unwrap());
    barcode.add_interval(PersistenceInterval::new(1.0, 3.0, 1).unwrap());

    assert_eq!(barcode.filter_by_dimension(0).len(), 1);
    assert_eq!(barcode.filter_by_dimension(1).len(), 2);
}

#[test]
#[verified_engine::verified]
fn test_barcode_filter_by_persistence() {
    let mut barcode = PersistenceBarcode::new();
    barcode.add_interval(PersistenceInterval::new(0.0, 0.5, 0).unwrap());
    barcode.add_interval(PersistenceInterval::new(0.5, 2.0, 1).unwrap());

    let significant = barcode.filter_by_persistence(1.0);
    assert_eq!(significant.len(), 1);
    assert_eq!(significant[0].dimension, 1);
}

#[test]
#[verified_engine::verified]
fn test_barcode_most_persistent() {
    let mut barcode = PersistenceBarcode::new();
    barcode.add_interval(PersistenceInterval::new(0.0, 1.0, 1).unwrap());
    barcode.add_interval(PersistenceInterval::new(0.5, 3.0, 1).unwrap());
    barcode.add_interval(PersistenceInterval::new(1.0, 2.0, 1).unwrap());

    let most_persistent = barcode.most_persistent(1).unwrap();
    assert!((most_persistent.persistence() - 2.5).abs() < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_compute_persistence_line() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(2.0, 0.0),
    ];
    let cloud = PointCloud::new(points).unwrap();

    let radii: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
    let barcode = compute_persistence(&cloud, &radii).unwrap();

    assert!(!barcode.is_empty());
    let holes = barcode.filter_by_dimension(1);
    let significant_holes: Vec<_> = holes.iter().filter(|h| h.is_significant(0.5)).collect();
    assert_eq!(significant_holes.len(), 0);
}

#[test]
#[verified_engine::verified]
fn test_compute_persistence_triangle() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(2.0, 0.0),
        Point2D::new(1.0, 1.732),
    ];
    let cloud = PointCloud::new(points).unwrap();

    let radii: Vec<f64> = (0..30).map(|i| i as f64 * 0.15).collect();
    let barcode = compute_persistence(&cloud, &radii).unwrap();
    assert!(!barcode.is_empty());
}

#[test]
#[verified_engine::verified]
fn test_compute_persistence_two_clusters() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(10.0, 0.0),
        Point2D::new(11.0, 0.0),
    ];
    let cloud = PointCloud::new(points).unwrap();

    let radii: Vec<f64> = (0..50).map(|i| i as f64 * 0.3).collect();
    let barcode = compute_persistence(&cloud, &radii).unwrap();

    let components = barcode.filter_by_dimension(0);
    let significant_components: Vec<_> = components
        .iter()
        .filter(|c| c.is_significant(1.0))
        .collect();
    assert!(!significant_components.is_empty());
}

#[test]
#[verified_engine::verified]
fn test_compute_persistence_invalid_radius() {
    let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0)];
    let cloud = PointCloud::new(points).unwrap();

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
}

#[test]
#[verified_engine::verified]
fn test_compute_persistence_barcode_parity() {
    let points = vec![
        Point2D::new(0.0, 0.0),
        Point2D::new(1.0, 0.0),
        Point2D::new(0.0, 1.0),
        Point2D::new(1.0, 1.0),
    ];
    let cloud = PointCloud::new(points).unwrap();
    let radii = [0.0, 0.5, 1.0, 1.5, 2.0];
    let barcode = compute_persistence(&cloud, &radii).unwrap();

    let mut b0_sequence = Vec::new();
    for &r in &radii {
        let c = vietoris_rips_complex(&cloud, r).unwrap();
        b0_sequence.push(betti_number_0(&c).unwrap());
    }
    assert_eq!(b0_sequence, vec![4, 4, 1, 1, 1]);
    assert_eq!(barcode.intervals.len(), 5);
}

#[verified_engine::verified]
fn bridged_loops_cloud() -> PointCloud {
    // Loop A (rhombus): sides = 1.05, short diagonal = 1.40 -> exact H1 [1.05, 1.40]
    let a = 0.70_f64;
    let b = (1.05_f64 * 1.05 - a * a).sqrt();

    // Loop B (symmetric 6-gon): sides <= 1.25 (closing at 1.25), vertical chord = 2.15 -> exact H1 [1.25, 2.15]
    // Bridged from Loop A's rightmost vertex (0.70, 0.0) to Loop B's leftmost vertex at distance 0.90.
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

    // Sampled compute_persistence must also preserve [1.05, 1.40] and [1.25, 2.15] without LIFO swapping
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
    // Loop 1: unit square of side 0.8 -> born at 0.8, dies at 0.8 * sqrt(2) ≈ 1.1314
    // Loop 2: well-separated square of side 1.3 -> born at 1.3, dies at 1.3 * sqrt(2) ≈ 1.8385
    // Coarse grid step from 1.0 to 1.5 sees beta_1(1.0) == 1 and beta_1(1.5) == 1 (net delta 0!),
    // which caused aggregate Betti tracking to miss Loop 1's death and Loop 2's birth.
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

    // Embed a square in R^4 on coordinates (w=0, x, y, z=5) and check H_1 interval [1.0, sqrt(2)]
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
    // 6-point cloud has C(6, 3) = 20 triangles; capping at 5 must report truncated_triangles = true
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
