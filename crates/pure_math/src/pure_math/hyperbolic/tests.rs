// [cite:pure_math]
#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn test_disk_and_half_plane_point_validation_and_clamping() {
    assert!(DiskPoint::new(0.3, -0.4).is_ok());
    assert!(DiskPoint::new(1.0, 0.0).is_err());
    assert!(DiskPoint::new(0.8, 0.8).is_err());
    assert!(DiskPoint::new(f64::NAN, 0.0).is_err());

    let clamped = DiskPoint::new_clamped(3.0, 4.0);
    assert!((clamped.modulus() - MAX_DISK_MODULUS).abs() < 1.0e-12);

    assert!(HalfPlanePoint::new(-2.5, 0.75).is_ok());
    assert!(HalfPlanePoint::new(0.0, 0.0).is_err());
    assert!(HalfPlanePoint::new(0.0, -0.1).is_err());
}

#[test]
fn test_cayley_roundtrip_and_metric_isometry() {
    let samples = [
        (0.0, 0.0),
        (0.25, -0.35),
        (-0.6, 0.45),
        (0.72, 0.51),
        (-0.15, -0.82),
    ];
    for i in 0..samples.len() {
        let p1 = DiskPoint::new(samples[i].0, samples[i].1).expect("valid disk point");
        let hp1 = cayley_disk_to_half_plane(p1);
        let back = cayley_half_plane_to_disk(hp1);
        assert!((p1.re() - back.re()).abs() < 1.0e-12);
        assert!((p1.im() - back.im()).abs() < 1.0e-12);

        for sample_j in samples.iter().skip(i + 1) {
            let p2 = DiskPoint::new(sample_j.0, sample_j.1).expect("valid disk point");
            let hp2 = cayley_disk_to_half_plane(p2);
            let d_disk = hyperbolic_distance_disk(p1, p2);
            let d_hp = hyperbolic_distance_half_plane(hp1, hp2);
            assert!(
                (d_disk - d_hp).abs() < 1.0e-11,
                "Cayley map must preserve hyperbolic distance: {d_disk} vs {d_hp}"
            );
        }
    }
}

#[test]
fn test_psu11_and_psl2r_distance_invariance_and_cayley_conjugation() {
    let p1 = DiskPoint::new(0.18, -0.27).expect("valid p1");
    let p2 = DiskPoint::new(-0.44, 0.39).expect("valid p2");
    let base_dist = hyperbolic_distance_disk(p1, p2);

    let drag = Psu11::drag_between(
        DiskPoint::new(-0.2, 0.1).expect("from"),
        DiskPoint::new(0.45, -0.35).expect("to"),
    );
    let iso = drag.compose(Psu11::rotation(1.15));
    assert!((iso.determinant() - 1.0).abs() < 1.0e-12);

    let moved_dist = hyperbolic_distance_disk(iso.apply(p1), iso.apply(p2));
    assert!((base_dist - moved_dist).abs() < 1.0e-11);

    let inv_check = iso.inverse().compose(iso);
    let p1_round = inv_check.apply(p1);
    assert!((p1.re() - p1_round.re()).abs() < 1.0e-12);
    assert!((p1.im() - p1_round.im()).abs() < 1.0e-12);

    let psl = iso.to_psl2r();
    assert!((psl.determinant() - 1.0).abs() < 1.0e-11);
    let hp_direct = cayley_disk_to_half_plane(iso.apply(p1));
    let hp_conj = psl.apply(cayley_disk_to_half_plane(p1));
    assert!((hp_direct.re() - hp_conj.re()).abs() < 1.0e-10);
    assert!((hp_direct.im() - hp_conj.im()).abs() < 1.0e-10);

    let psu_back = psl.to_psu11();
    let p2_via_psl = psu_back.apply(p2);
    let p2_direct = iso.apply(p2);
    assert!((p2_via_psl.re() - p2_direct.re()).abs() < 1.0e-11);
    assert!((p2_via_psl.im() - p2_direct.im()).abs() < 1.0e-11);
}

#[test]
fn test_schlafli_validation_rejects_spherical_and_euclidean_pairs() {
    let invalid_pairs = [
        (2, 5),
        (5, 2),
        (3, 3),
        (4, 3),
        (3, 4),
        (5, 3),
        (3, 5),
        (4, 4),
        (6, 3),
        (3, 6),
    ];
    for (p, q) in invalid_pairs {
        assert_eq!(
            validate_hyperbolic_schlafli(p, q),
            Err(HyperbolicError::NonHyperbolicSchlafli { p, q })
        );
    }

    let valid_pairs = [(7, 3), (3, 7), (5, 4), (4, 5), (8, 3), (6, 4), (8, 8)];
    for (p, q) in valid_pairs {
        assert!(validate_hyperbolic_schlafli(p, q).is_ok());
    }
}

#[test]
fn test_gauss_bonnet_polygon_areas_and_isometry_invariance() {
    let cases = [(7u32, 3u32), (5, 4), (8, 3), (6, 4)];
    let iso = Psu11::drag_between(
        DiskPoint::origin(),
        DiskPoint::new(0.38, -0.29).expect("interior"),
    )
    .compose(Psu11::rotation(0.7));

    for (p, q) in cases {
        let verts = regular_polygon_vertices(p, q).expect("valid {p,q}");
        let expected_area = (f64::from(p - 2) - 2.0 * f64::from(p) / f64::from(q)) * PI;
        let area = gauss_bonnet_polygon_area(&verts).expect("area");
        assert!(
            (area - expected_area).abs() < 1.0e-11,
            "Gauss-Bonnet mismatch for {{{p},{q}}}: got {area}, expected {expected_area}"
        );

        let moved_verts: Vec<Complex64> = verts.iter().map(|&z| iso.apply_complex(z)).collect();
        let moved_area = gauss_bonnet_polygon_area(&moved_verts).expect("moved area");
        assert!((moved_area - expected_area).abs() < 1.0e-10);
    }

    let ideal_tri = ideal_polygon_vertices(3).expect("ideal triangle");
    let tri_area = gauss_bonnet_polygon_area(&ideal_tri).expect("ideal triangle area");
    assert!((tri_area - PI).abs() < 1.0e-11);

    let ideal_quad = ideal_polygon_vertices(4).expect("ideal quad");
    let quad_area = gauss_bonnet_polygon_area(&ideal_quad).expect("ideal quad area");
    assert!((quad_area - 2.0 * PI).abs() < 1.0e-11);
}

#[test]
fn test_hyperbolic_tiling_generation_regular_and_ideal() {
    let buf_73 =
        HyperbolicTilingBuffer::new(TilingKind::Regular { p: 7, q: 3 }, 3).expect("{7,3} tiling");
    assert!(buf_73.polygons().len() > 20);
    assert!(buf_73.polygons().len() <= MAX_TILING_POLYGONS);

    let expected_hept_area = PI / 3.0;
    for poly in buf_73.polygons().iter().take(15) {
        let area = gauss_bonnet_polygon_area(poly.active_vertices()).expect("tile area");
        assert!((area - expected_hept_area).abs() < 1.0e-9);
    }

    let buf_inf3 =
        HyperbolicTilingBuffer::new(TilingKind::Ideal { q: 3 }, 3).expect("{infinity,3} tiling");
    assert!(buf_inf3.polygons().len() > 10);
    assert!(buf_inf3.polygons().len() <= MAX_TILING_POLYGONS);
}

#[test]
fn test_horocycle_construction_and_half_plane_mapping() {
    let through = DiskPoint::new(0.2, 0.1).expect("through");
    let h_finite = HorocycleDisk::through_point(Complex64::new(0.0, 1.0), through).expect("horo");
    match h_finite.to_half_plane() {
        HorocycleHalfPlane::TangentCircle { base_x, radius } => {
            assert!((base_x + 1.0).abs() < 1.0e-11);
            assert!(radius > 0.0);
        }
        HorocycleHalfPlane::HorizontalLine { .. } => panic!("expected tangent circle for xi = i"),
    }

    let h_inf = HorocycleDisk::from_radius(Complex64::new(1.0, 0.0), 0.25).expect("horo at 1");
    match h_inf.to_half_plane() {
        HorocycleHalfPlane::HorizontalLine { height } => assert!(height > 0.0),
        HorocycleHalfPlane::TangentCircle { .. } => panic!("expected horizontal line for xi = 1"),
    }
}

#[test]
fn test_chaotic_twin_billiards_divergence_and_symbolic_dynamics() {
    let mut sys = TwinBilliardSystem::new(
        BilliardDomainKind::RegularPolygon { p: 8, q: 3 },
        42,
        1.0e-6,
    )
    .expect("octagon billiard");
    assert!(sys.current_log_separation().abs() < 1.0e-11);

    sys.advance_reflections(25);
    assert!(sys.total_bounces() >= 25);
    assert!(
        sys.current_log_separation() > 2.0,
        "twin rays in negative curvature must diverge exponentially"
    );

    let mut codes = [0u8; 32];
    let n_codes = sys.copy_edge_hits_into(&mut codes);
    assert!(n_codes >= 25);
    assert!(codes[..n_codes].iter().all(|&edge| edge < 8));

    let mut ideal_sys =
        TwinBilliardSystem::new(BilliardDomainKind::IdealPolygon { sides: 3 }, 7, 1.0e-6)
            .expect("ideal triangle billiard");
    ideal_sys.advance_reflections(20);
    assert!(ideal_sys.total_bounces() >= 20);
    assert!(ideal_sys.current_log_separation() > 1.5);
}
