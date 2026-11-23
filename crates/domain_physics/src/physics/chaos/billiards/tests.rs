//! Physics conservation and dispersion tests for flat Euclidean billiards.

use super::*;

#[test]
#[verified_engine::verified]
fn test_circular_billiard_conserves_angular_momentum_and_unit_speed_over_1000_bounces() {
    let circle = CircularBilliard::new(1.25);
    let mut ray = BilliardRay::from_angle([0.35, -0.2], 0.61).expect("valid ray");
    let lz0 = circle.angular_momentum(&ray);
    let mut first_sin_phi = None;

    for bounce in 0..1_000 {
        let hit = circle
            .collide(&mut ray)
            .unwrap_or_else(|| panic!("circle collision {bounce} failed"));
        assert!((0.0..1.0).contains(&hit.arc_fraction));
        assert!((-1.0..=1.0).contains(&hit.sin_phi));
        assert!((ray.speed() - 1.0).abs() < 1e-14);
        assert!(
            (circle.angular_momentum(&ray) - lz0).abs() < 1e-12,
            "Circle angular momentum drifted at bounce {bounce}"
        );
        let sp = *first_sin_phi.get_or_insert(hit.sin_phi);
        assert!(
            (hit.sin_phi - sp).abs() < 1e-12,
            "Circle Birkhoff sin(phi) must stay constant"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_elliptical_billiard_conserves_focal_product_and_unit_speed_over_1000_bounces() {
    let ellipse = EllipticalBilliard::new(1.5, 1.0);

    // Case 1: Rotational orbit (I > 0, confocal ellipse caustic)
    let mut rot_ray = BilliardRay::from_angle([0.0, 0.72], 0.18).expect("rotational ray");
    let i_rot_0 = ellipse.focal_angular_momentum_product(&rot_ray);
    assert!(i_rot_0 > 0.0);

    // Case 2: Librational orbit crossing between foci (I < 0, confocal hyperbola caustic)
    let mut lib_ray = BilliardRay::from_angle([0.0, 0.0], 1.15).expect("librational ray");
    let i_lib_0 = ellipse.focal_angular_momentum_product(&lib_ray);
    assert!(i_lib_0 < 0.0);

    for bounce in 0..1_000 {
        let h_rot = ellipse
            .collide(&mut rot_ray)
            .unwrap_or_else(|| panic!("ellipse rotational bounce {bounce} failed"));
        let h_lib = ellipse
            .collide(&mut lib_ray)
            .unwrap_or_else(|| panic!("ellipse librational bounce {bounce} failed"));
        assert!((0.0..1.0).contains(&h_rot.arc_fraction));
        assert!((0.0..1.0).contains(&h_lib.arc_fraction));
        assert!((rot_ray.speed() - 1.0).abs() < 1e-14);
        assert!((lib_ray.speed() - 1.0).abs() < 1e-14);
        assert!(
            (ellipse.focal_angular_momentum_product(&rot_ray) - i_rot_0).abs() < 1e-11,
            "Rotational focal product drifted at bounce {bounce}"
        );
        assert!(
            (ellipse.focal_angular_momentum_product(&lib_ray) - i_lib_0).abs() < 1e-11,
            "Librational focal product drifted at bounce {bounce}"
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_stadium_and_sinai_1000_collisions_unit_speed_and_ray_fan_divergence() {
    let stadium = BilliardTable::Stadium(StadiumBilliard::new(0.85, 1.0));
    let sinai = BilliardTable::Sinai(SinaiBilliard::new(1.0, 0.36));

    // Invalid seed inside Sinai central scatterer must be rejected cleanly
    let mut bad_fan = [BilliardRay {
        pos: [0.0, 0.0],
        vel: [1.0, 0.0],
    }; 4];
    assert_eq!(sinai.seed_ray_fan([0.1, 0.1], 0.4, 0.02, &mut bad_fan), 0);

    let mut st_fan = [BilliardRay {
        pos: [0.0, 0.0],
        vel: [1.0, 0.0],
    }; 2];
    let mut si_fan = [BilliardRay {
        pos: [0.0, 0.0],
        vel: [1.0, 0.0],
    }; 2];
    assert_eq!(stadium.seed_ray_fan([0.1, 0.2], 0.47, 1e-5, &mut st_fan), 2);
    assert_eq!(sinai.seed_ray_fan([0.65, 0.2], 2.55, 1e-5, &mut si_fan), 2);

    for bounce in 0..1_000 {
        for r in &mut st_fan {
            let hit = stadium
                .collide(r)
                .unwrap_or_else(|| panic!("stadium bounce {bounce} failed"));
            assert!((0.0..1.0).contains(&hit.arc_fraction));
            assert!((-1.0..=1.0).contains(&hit.sin_phi));
            assert!((r.speed() - 1.0).abs() < 1e-14);
        }
        for r in &mut si_fan {
            let hit = sinai
                .collide(r)
                .unwrap_or_else(|| panic!("sinai bounce {bounce} failed"));
            assert!((0.0..1.0).contains(&hit.arc_fraction));
            assert!((-1.0..=1.0).contains(&hit.sin_phi));
            assert!((r.speed() - 1.0).abs() < 1e-14);
        }
    }

    let st_sep = (st_fan[0].vel[0] - st_fan[1].vel[0]).hypot(st_fan[0].vel[1] - st_fan[1].vel[1]);
    let si_sep = (si_fan[0].vel[0] - si_fan[1].vel[0]).hypot(si_fan[0].vel[1] - si_fan[1].vel[1]);
    assert!(
        st_sep > 0.05,
        "Bunimovich stadium rays must defocus: {st_sep}"
    );
    assert!(
        si_sep > 0.05,
        "Sinai dispersing rays must diverge: {si_sep}"
    );
}
