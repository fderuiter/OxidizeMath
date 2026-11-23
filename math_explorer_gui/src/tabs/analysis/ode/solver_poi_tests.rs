#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_harmonic_equilibrium_root_dt_025_and_nonlinear_preset_fits() {
    let mut osc = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1.0,
        ic_y0: 1.0,
        ic_v0: 0.0,
        dt: 0.25,
        total_time: 10.0,
        ..Default::default()
    };
    osc.recalculate();
    let root_t = annotations::harmonic_equilibrium_root(&osc).expect("harmonic root");
    assert!(
        (root_t - 5.0).abs() > 0.1,
        "must not anchor at arbitrary midpoint t=5.0"
    );
    let nearest_analytical = 1.5 * std::f64::consts::PI;
    assert!(
        (root_t - nearest_analytical).abs() < 1e-3,
        "dt=0.25 RK4 + Hermite root error = {:.6e}s",
        (root_t - nearest_analytical).abs()
    );

    let osc_series = (&osc.time_series[..], &osc.y_series[..]);
    let sin_fit =
        math_explorer::fit::fit_model(&osc_series, math_explorer::fit::FitModel::Sinusoidal, false)
            .expect("sinusoidal fit");
    assert!(sin_fit.r_squared >= 0.99);

    let mut log_tool = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: 0.8,
        param_cap_k: 12.0,
        ic_y0: 0.5,
        dt: 0.1,
        total_time: 12.0,
        ..Default::default()
    };
    log_tool.recalculate();
    let log_series = (&log_tool.time_series[..], &log_tool.y_series[..]);
    let logistic_fit =
        math_explorer::fit::fit_model(&log_series, math_explorer::fit::FitModel::Logistic, false)
            .expect("logistic fit");
    assert!(logistic_fit.r_squared >= 0.99);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            let r = annotations::harmonic_equilibrium_root(&osc);
            assert!(r.is_some());
        });
    }
}

#[test]
fn test_export_promoted_curve_solver_fits_png() {
    let mut osc = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1.0,
        ic_y0: 1.0,
        ic_v0: 0.0,
        dt: 0.25,
        total_time: 10.0,
        ..Default::default()
    };
    osc.recalculate();
    let series = (&osc.time_series[..], &osc.y_series[..]);
    let fit =
        math_explorer::fit::fit_model(&series, math_explorer::fit::FitModel::Sinusoidal, false)
            .expect("sinusoidal fit");
    let raw: Vec<[f64; 2]> = osc
        .time_series
        .iter()
        .copied()
        .zip(osc.y_series.iter().copied())
        .map(|(t, y)| [t, y])
        .collect();
    let fitted: Vec<[f64; 2]> = (0..=200)
        .map(|k| {
            let t = (k as f64) * 0.05;
            [t, fit.eval(t)]
        })
        .collect();
    let root_t = annotations::harmonic_equilibrium_root(&osc).expect("root");
    let datasets = vec![
        ("RK4 Harmonic Trajectory (dt=0.25)".to_string(), raw),
        (
            format!("math_commons Sinusoidal Fit ({})", fit.format_formula()),
            fitted,
        ),
        ("Equilibrium y=0".to_string(), vec![[0.0, 0.0], [10.0, 0.0]]),
        (
            format!("Hermite Equilibrium Root (t={root_t:.4})"),
            vec![[root_t, -0.08], [root_t, 0.08]],
        ),
    ];
    let png = egui_plot::export::export_custom_png(
        &datasets,
        &egui_plot::export::ExportOptions::default(),
    )
    .expect("export sinusoidal fit png");
    assert!(!png.is_empty());
    let _ = std::fs::write("/tmp/math_commons_ode_sinusoidal_fit_poi.png", png);
}
