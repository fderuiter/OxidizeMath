// [cite:advanced_linear_algebra]

use std::f64::consts::PI;

use math_commons::math_kernel::types::flatten_2d_index;

use super::*;

#[verified_engine::verified]
fn measure_drumhead_frequency_128(mode: &Eigenmode2D, params: &Wave2DParams) -> f64 {
    let n = 128_usize;
    let mut solver = Wave2DSolver::new_unit_square(n);
    apply_circular_drumhead_mask(&mut solver.wall_mask, n, n, 0.45);
    let mut init = vec![0.0; n * n];
    let denom = (n - 1) as f64;
    for y in 0..n {
        let yn = (y as f64) / denom;
        for x in 0..n {
            let idx = flatten_2d_index(x, y, n);
            if solver.wall_mask[idx] > 0.5 {
                let xn = (x as f64) / denom;
                init[idx] = evaluate_eigenmode_amplitude(mode, xn, yn, 0.45);
            }
        }
    }
    solver.set_stationary_displacement(&init);
    let (_sub, dt_sub) = solver.safe_substep_plan(params, 1.0);
    let dt = dt_sub * 0.8;
    let mut prev_proj: f64 = init.iter().map(|v| v * v).sum();
    let mut t_cross = 0.0_f64;
    for step_idx in 0..400 {
        solver.step(params, dt);
        let curr_proj: f64 = solver
            .displacement()
            .iter()
            .zip(init.iter())
            .map(|(u, phi)| u * phi)
            .sum();
        if prev_proj > 0.0 && curr_proj <= 0.0 {
            let frac = prev_proj / (prev_proj - curr_proj);
            t_cross = ((step_idx as f64) + frac) * dt;
            break;
        }
        prev_proj = curr_proj;
    }
    assert!(t_cross > 0.0, "Drumhead mode failed to reach zero crossing");
    0.25 / t_cross
}

#[verified_engine::verified]
fn measure_plate_rayleigh_frequency_128(mode: &Eigenmode2D, params: &Wave2DParams) -> f64 {
    let n = 128_usize;
    let mut solver = Wave2DSolver::new_unit_square(n);
    apply_point_clamp(&mut solver.wall_mask, n, n, 0.5, 0.5, 2.2);
    let mut init = vec![0.0; n * n];
    let denom = (n - 1) as f64;
    for y in 0..n {
        let yn = (y as f64) / denom;
        for x in 0..n {
            let idx = flatten_2d_index(x, y, n);
            if solver.wall_mask[idx] > 0.5 {
                let xn = (x as f64) / denom;
                init[idx] = evaluate_eigenmode_amplitude(mode, xn, yn, 0.45);
            }
        }
    }
    solver.set_stationary_displacement(&init);
    let (_sub, dt_sub) = solver.safe_substep_plan(params, 1.0);
    let dt = dt_sub * 0.25;
    solver.step(params, dt);
    let mut numer = 0.0_f64;
    let mut denom_norm = 0.0_f64;
    for (&u1, &u0) in solver.displacement().iter().zip(init.iter()) {
        numer += (u0 - u1) * u0;
        denom_norm += u0 * u0;
    }
    let omega_sq = (2.0 * numer) / (dt * dt * denom_norm);
    omega_sq.max(0.0).sqrt() / (2.0 * PI)
}

#[test]
#[verified_engine::verified]
fn test_neumann_chladni_and_bessel_drumhead_discrete_eigenfrequencies() {
    let plate_params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Neumann,
        wave_speed: 0.0,
        plate_stiffness: 0.0004,
        damping: 0.0,
    };
    let mut modes = [Eigenmode2D::default(); 24];
    let count = populate_eigenmodes(
        false,
        128,
        1.0,
        0.45,
        &plate_params,
        PinConstraint2D::CenterPin,
        &mut modes,
    );
    assert!(count >= 4);
    for mode in modes.iter().take(4) {
        let measured = measure_plate_rayleigh_frequency_128(mode, &plate_params);
        let err = (measured - mode.frequency_hz).abs();
        assert!(
            err < 0.001,
            "Plate mode ({},{}) {:?} measured={measured:.6} Hz vs predicted={:.6} Hz (err={err:.6})",
            mode.m,
            mode.n,
            mode.kind,
            mode.frequency_hz
        );
    }

    let drum_params = Wave2DParams {
        medium: Wave2DMedium::AcousticWave,
        boundary: BoundaryMode2D::Dirichlet,
        wave_speed: 1.0,
        plate_stiffness: 0.0,
        damping: 0.0,
    };
    let d_count = populate_eigenmodes(
        true,
        128,
        1.0,
        0.45,
        &drum_params,
        PinConstraint2D::None,
        &mut modes,
    );
    assert!(d_count >= 6);
    for mode in modes.iter().take(4) {
        let measured = measure_drumhead_frequency_128(mode, &drum_params);
        let err = (measured - mode.frequency_hz).abs();
        assert!(
            err < 0.002,
            "Drumhead mode ({},{}) measured={measured:.6} Hz vs predicted={:.6} Hz (err={err:.6})",
            mode.m,
            mode.n,
            mode.frequency_hz
        );
    }
}

#[test]
fn test_resonant_antinodal_lock_boosts_steady_state_energy_over_50x() {
    let n = 17_usize;
    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Dirichlet,
        wave_speed: 0.0,
        plate_stiffness: 1.0,
        damping: 0.18,
    };
    let mut modes = [Eigenmode2D::default(); 24];
    let count = populate_eigenmodes(
        false,
        n,
        1.0,
        0.45,
        &params,
        PinConstraint2D::CenterPin,
        &mut modes,
    );
    assert!(count >= 2);
    let target = modes[0];
    let (ax, ay) = snap_to_mode_antinode(&target, 0.22, 0.50, PinConstraint2D::CenterPin, 0.45);

    let mut off_solver = Wave2DSolver::new_unit_square(n);
    apply_point_clamp(&mut off_solver.wall_mask, n, n, 0.5, 0.5, 0.6);
    let (_, dt_lim) = off_solver.safe_substep_plan(&params, 1.0);
    let dt_macro = dt_lim * 60.0;
    let off_driver = DriverConfig2D {
        enabled: true,
        line_source: false,
        x_norm: 0.22,
        y_norm: 0.50,
        frequency: target.frequency_hz * 0.42,
        amplitude: 65.0,
        radius_cells: 1.0,
    };

    let mut res_solver = Wave2DSolver::new_unit_square(n);
    apply_point_clamp(&mut res_solver.wall_mask, n, n, 0.5, 0.5, 0.6);
    let res_driver = DriverConfig2D {
        x_norm: ax,
        y_norm: ay,
        frequency: target.frequency_hz,
        ..off_driver
    };

    let mut off_mean_energy = 0.0_f64;
    let mut res_mean_energy = 0.0_f64;
    for step_idx in 0..300 {
        off_solver.step_driven(&params, &off_driver, dt_macro, None);
        res_solver.step_driven(&params, &res_driver, dt_macro, None);
        if step_idx >= 260 {
            off_mean_energy += off_solver.displacement().iter().map(|v| v * v).sum::<f64>();
            res_mean_energy += res_solver.displacement().iter().map(|v| v * v).sum::<f64>();
        }
    }
    assert!(
        res_mean_energy > 50.0 * off_mean_energy.max(1.0e-12),
        "Expected >50x steady-state energy gain, got {:.2}x (res={res_mean_energy:.4e}, off={off_mean_energy:.4e})",
        res_mean_energy / off_mean_energy.max(1.0e-12)
    );
}

#[test]
fn test_eigenmodes_and_lorentzian_spectrum_zero_allocations() {
    let params = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: BoundaryMode2D::Neumann,
        wave_speed: 0.0,
        plate_stiffness: 0.0004,
        damping: 0.02,
    };
    let (resp, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut buf = [Eigenmode2D::default(); 24];
        let count = populate_eigenmodes(
            false,
            128,
            1.0,
            0.45,
            &params,
            PinConstraint2D::CenterPin,
            &mut buf,
        );
        let (ax, ay) = snap_to_mode_antinode(&buf[0], 0.22, 0.50, PinConstraint2D::CenterPin, 0.45);
        let (r, _) = lorentzian_modal_response(
            &buf[..count],
            buf[0].frequency_hz,
            0.02,
            Some((ax, ay)),
            0.45,
        );
        r
    });
    assert!(resp > 100.0);
    assert!(metrics.function_calls > 0);
}

#[cfg(feature = "serde")]
#[test]
fn test_eigenmode_2d_serde_roundtrip() {
    let mode = Eigenmode2D {
        m: 2,
        n: 3,
        frequency_hz: 440.0,
        omega: 880.0 * std::f64::consts::PI,
        wavenumber: 11.2,
        antinode_x: 0.25,
        antinode_y: 0.75,
        kind: PlateEigenmodeKind::SquareChladniAntisymmetric,
    };
    let json = serde_json::to_string(&(PinConstraint2D::QuadSymmetry, mode)).expect("serialize");
    let restored: (PinConstraint2D, Eigenmode2D) =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, (PinConstraint2D::QuadSymmetry, mode));
}
