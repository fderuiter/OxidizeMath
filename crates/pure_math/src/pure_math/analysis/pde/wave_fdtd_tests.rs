#![cfg_attr(any(), verified(opt_out = "tests"))]
// [cite:optics]

use super::*;
use math_commons::math_kernel::types::flatten_2d_index;

#[test]
fn test_wave_field_2d_zero_heap_allocation_under_verified_allocator() {
    let config = WaveFieldConfig2D {
        width: 64,
        height: 64,
        dx: 1.0 / 64.0,
        dy: 1.0 / 64.0,
        ..WaveFieldConfig2D::default()
    };
    let mut field = WaveField2D::new(config);
    let src = WaveSource2D {
        kind: WaveSourceKind::PlaneWave {
            x_index: 12,
            y_min: 14,
            y_max: 50,
        },
        frequency: 10.0,
        amplitude: 40.0,
        phase: 0.0,
    };
    let dt = field.cfl_dt();

    let (finite, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..20 {
            field.step(dt, Some(&src));
        }
        field.is_finite()
    });
    assert!(finite);
    assert!(metrics.function_calls > 0);

    // Also verify negative-index metamaterial mode runs with zero dynamic heap allocation.
    field.set_reference_frequency(10.0);
    field.configure_medium_in_place(|x, _y, _cfg, n_out, obs_out| {
        *obs_out = 0;
        *n_out = if (26..=38).contains(&x) { -1.0 } else { 1.0 };
    });
    assert!(field.has_negative_index());
    let drude_dt = field.cfl_dt();
    let (finite_neg, metrics_neg) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..20 {
            field.step(drude_dt, Some(&src));
        }
        field.is_finite()
    });
    assert!(finite_neg);
    assert!(metrics_neg.function_calls > 0);
}

#[test]
fn test_cfl_clamping_and_sponge_absorption() {
    let config = WaveFieldConfig2D {
        width: 80,
        height: 80,
        dx: 1.0 / 80.0,
        dy: 1.0 / 80.0,
        sponge_width: 12,
        sponge_max_damping: 30.0,
        ..WaveFieldConfig2D::default()
    };
    let mut field = WaveField2D::new(config);
    let cfl = field.cfl_dt();
    // Overly large or invalid dt is clamped to cfl_dt.
    assert!((field.clamp_dt(10.0) - cfl).abs() < 1e-12);
    assert!((field.clamp_dt(f64::NAN) - cfl).abs() < 1e-12);

    let pulse_src = WaveSource2D {
        kind: WaveSourceKind::PointSource { x: 40, y: 40 },
        frequency: 12.0,
        amplitude: 80.0,
        phase: 0.0,
    };
    for _ in 0..40 {
        field.step(1.0, Some(&pulse_src));
    }
    let peak_energy: f64 = field.field().iter().map(|u| u * u).sum();
    assert!(peak_energy > 1e-3);

    // Let pulse propagate into the sponge layer without driving source.
    for _ in 0..220 {
        field.step(cfl, None);
    }
    let residual_energy: f64 = field.field().iter().map(|u| u * u).sum();
    assert!(
        residual_energy < 0.08 * peak_energy,
        "sponge boundary must absorb outgoing pulse energy ({residual_energy} vs {peak_energy})"
    );
}

fn build_double_slit_field(n: usize, wall_x: usize, slit_sep_cells: usize) -> WaveField2D {
    let dx = 1.0 / (n as f64);
    let mut field = WaveField2D::new(WaveFieldConfig2D {
        width: n,
        height: n,
        dx,
        dy: dx,
        c0: 1.0,
        sponge_width: 10,
        sponge_max_damping: 26.0,
        irradiance_decay: 0.02,
    });
    let center_y = n / 2;
    let slit1_y = center_y - slit_sep_cells / 2;
    let slit2_y = center_y + slit_sep_cells / 2;
    field.configure_medium_in_place(|x, y, _cfg, n_out, obs_out| {
        *n_out = 1.0;
        if x == wall_x {
            let open = y.abs_diff(slit1_y) <= 1 || y.abs_diff(slit2_y) <= 1;
            *obs_out = u8::from(!open);
        } else {
            *obs_out = 0;
        }
    });
    field
}

#[test]
fn test_double_slit_central_and_first_order_fringes() {
    let n = 128;
    let dx = 1.0 / (n as f64);
    let wall_x = 34_usize;
    let slit_sep_cells = 20_usize;
    let center_y = n / 2;
    let mut field = build_double_slit_field(n, wall_x, slit_sep_cells);

    let freq = 16.0;
    let src = WaveSource2D {
        kind: WaveSourceKind::PlaneWave {
            x_index: 16,
            y_min: 14,
            y_max: n - 14,
        },
        frequency: freq,
        amplitude: 120.0,
        phase: 0.0,
    };
    let dt = field.cfl_dt();
    for _ in 0..380 {
        field.step(dt, Some(&src));
    }

    let det_x = 90_usize;
    let screen_dist = ((det_x - wall_x) as f64) * dx;
    let slit_sep = (slit_sep_cells as f64) * dx;
    let lambda = 1.0 / freq;
    let expected_dy = lambda
        * ((screen_dist * screen_dist) / (slit_sep * slit_sep - lambda * lambda) + 0.25).sqrt();

    let irr = field.irradiance();
    let center_irr = irr[flatten_2d_index(det_x, center_y, n)];
    let best_y = ((center_y + 8)..(center_y + 36))
        .max_by(|&a, &b| {
            irr[flatten_2d_index(det_x, a, n)].total_cmp(&irr[flatten_2d_index(det_x, b, n)])
        })
        .unwrap_or(center_y + 8);
    let measured_dy = ((best_y - center_y) as f64) * dx;
    let trough_y = center_y + ((0.5 * expected_dy / dx).round() as usize);
    let trough_irr = irr[flatten_2d_index(det_x, trough_y, n)];
    assert!(
        center_irr > 2.5 * trough_irr,
        "central bright fringe ({center_irr}) must exceed dark minimum ({trough_irr})"
    );
    assert!(
        (measured_dy - expected_dy).abs() < 0.04,
        "measured 1st fringe offset {measured_dy:.4} must match theoretical {expected_dy:.4}"
    );
}

#[test]
fn test_luneburg_lens_wave_focusing() {
    let n = 128;
    let dx = 1.0 / (n as f64);
    let config = WaveFieldConfig2D {
        width: n,
        height: n,
        dx,
        dy: dx,
        c0: 1.0,
        sponge_width: 10,
        sponge_max_damping: 24.0,
        irradiance_decay: 0.025,
    };
    let mut field = WaveField2D::new(config);
    let cx = 0.50;
    let cy = 0.50;
    let radius = 0.22;
    field.configure_medium_in_place(|gx, gy, cfg, n_out, obs_out| {
        *obs_out = 0;
        let x = (gx as f64 + 0.5) * cfg.dx;
        let y = (gy as f64 + 0.5) * cfg.dy;
        let r_sq = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (radius * radius);
        *n_out = if r_sq <= 1.0 {
            (2.0 - r_sq).sqrt()
        } else {
            1.0
        };
    });

    let src = WaveSource2D {
        kind: WaveSourceKind::PlaneWave {
            x_index: 16,
            y_min: 20,
            y_max: n - 20,
        },
        frequency: 14.0,
        amplitude: 90.0,
        phase: 0.0,
    };
    let dt = field.cfl_dt();
    for _ in 0..420 {
        field.step(dt, Some(&src));
    }

    // Rear rim of Luneburg lens is at (cx + radius, cy) = (0.72, 0.50).
    let focus_x = ((cx + radius) / dx).round() as usize;
    let focus_y = (cy / dx).round() as usize;
    let irr = field.irradiance();
    let focus_irr = irr[flatten_2d_index(focus_x, focus_y, n)];
    let off_axis_irr = irr[flatten_2d_index(focus_x, focus_y + 22, n)];
    assert!(
        focus_irr > 3.0 * off_axis_irr,
        "Luneburg rear-surface focus irradiance ({focus_irr}) must strongly exceed off-axis ({off_axis_irr})"
    );
}
