use super::*;
use math_commons::math_kernel::types::flatten_2d_index;

#[test]
#[verified_engine::verified]
fn test_spatial_kernel_unit_normalization_and_symmetry() {
    let mut kernel = vec![0.0; 64 * 64];
    build_spatial_kernel(
        64,
        64,
        13,
        &[1.0],
        LeniaKernelCore::ExponentialBump,
        &mut kernel,
    );
    let sum: f64 = kernel.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "kernel sum was {sum}");

    let k_pos = kernel[flatten_2d_index(6, 0, 64)];
    let k_neg = kernel[flatten_2d_index(64 - 6, 0, 64)];
    assert!((k_pos - k_neg).abs() < 1e-12);
    assert!(k_pos > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_orbium_translates_across_periodic_boundary_with_bounded_mass() {
    let width = 64;
    let height = 64;
    let params = LeniaPreset::Orbium.params();
    let mut system = LeniaSystem::new(width, height, params).expect("valid system");

    // Stamp Orbium near the Periodic boundary so its glide crosses wrap-around.
    LeniaPreset::Orbium.stamp_into_grid(
        system.grid_mut(),
        width,
        height,
        13,
        width - 8,
        height - 8,
    );
    system.recompute_telemetry();
    let initial_mass = system.telemetry().total_mass;
    let initial_com = system.telemetry().center_of_mass;
    assert!(initial_mass > 40.0, "initial mass={initial_mass}");

    let mut total_dx = 0.0;
    let mut total_dy = 0.0;
    let mut prev_com = initial_com;
    let mut max_speed = 0.0_f64;

    for _ in 0..220 {
        system.step();
        let tel = system.telemetry();
        assert!(tel.total_mass.is_finite());
        assert!(
            tel.total_mass > 25.0 && tel.total_mass < 320.0,
            "Orbium mass drifted out of bounds: {}",
            tel.total_mass
        );
        assert!(tel.speed.is_finite());
        // Velocity must remain smooth across periodic wrap-around (no grid-width jump).
        assert!(
            tel.speed < 25.0,
            "Unwrapped periodic velocity spiked to {}",
            tel.speed
        );
        max_speed = max_speed.max(tel.speed);
        total_dx += shortest_toroidal_delta(tel.center_of_mass[0], prev_com[0], width as f64);
        total_dy += shortest_toroidal_delta(tel.center_of_mass[1], prev_com[1], height as f64);
        prev_com = tel.center_of_mass;
    }

    let net_displacement = total_dx.hypot(total_dy);
    assert!(
        net_displacement > 10.0,
        "Orbium should translate across the torus, got displacement {net_displacement}"
    );
    assert!(max_speed > 0.2, "Orbium should exhibit nonzero glide speed");
}

#[test]
#[verified_engine::verified]
fn test_hydrogeminium_preset_remains_finite_and_active() {
    let mut system =
        LeniaSystem::from_preset(96, 96, LeniaPreset::Hydrogeminium).expect("valid hydrogeminium");
    for _ in 0..60 {
        system.step();
    }
    let tel = system.telemetry();
    assert!(tel.total_mass.is_finite());
    assert!(
        tel.total_mass > 10.0,
        "Hydrogeminium mass={}",
        tel.total_mass
    );
    assert!((0.0..=1.0).contains(&tel.rotational_symmetry));
    assert!((1..=6).contains(&tel.dominant_symmetry_fold));
}

#[test]
#[verified_engine::verified]
fn test_radial_disc_exhibits_high_rotational_symmetry() {
    let mut system = LeniaSystem::new(64, 64, LeniaParams::default()).expect("valid system");
    for y in 0..64 {
        for x in 0..64 {
            let d = ((x as f64) - 32.0).hypot((y as f64) - 32.0);
            if d <= 8.0 {
                let idx = flatten_2d_index(x, y, 64);
                system.grid_mut()[idx] = 1.0 - (d / 9.0);
            }
        }
    }
    system.recompute_telemetry();
    let tel = system.telemetry();
    assert!(
        tel.rotational_symmetry > 0.95,
        "Expected radially symmetric disc score > 0.95, got {}",
        tel.rotational_symmetry
    );
}
