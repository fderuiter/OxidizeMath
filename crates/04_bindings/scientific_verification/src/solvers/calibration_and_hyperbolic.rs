use domain_physics::physics::fluid_dynamics::potential_flow::{PotentialFlowField, Vortex};
use domain_physics::physics::fluid_dynamics::solver::PotentialTracerSystem;
use math_commons::primitives::UnitInterval;
use nalgebra::Vector2;
use pure_math::pure_math::analysis::ode::TimeStepper;
use pure_math::pure_math::hyperbolic::{BilliardDomainKind, DiskPoint, TwinBilliardSystem};
use pure_math::pure_math::statistics::calibration::{
    IsotonicCalibrator, PavaBlock, PlattConfig, PlattScaler, fit_pava_sorted_into,
};
use std::f64::consts::PI;
use verified_engine::engine::VerifiedEngine;

use crate::mms::{SolverCertificate, ols_convergence_order};

fn certify_potential_tracer_rk4() -> SolverCertificate {
    let mut field = PotentialFlowField::new();
    field.add(Box::new(Vortex::new(2.0 * PI, 0.0, 0.0)));
    let t_end = 0.8_f64;
    let exact = Vector2::new(t_end.cos(), t_end.sin());
    let dts = [0.1_f64, 0.05, 0.025, 0.0125];
    let mut errs = [0.0_f64; 4];

    for (idx, &dt) in dts.iter().enumerate() {
        let steps = (t_end / dt).round() as usize;
        let mut tracer = PotentialTracerSystem::new(&field, Vector2::new(1.0, 0.0), 10.0);
        let (_, metrics) = VerifiedEngine::run_verified(|| {
            for _ in 0..steps {
                tracer.step(dt);
            }
        });
        assert!(metrics.function_calls > 0);
        errs[idx] = (tracer.position - exact).norm();
    }

    let order = ols_convergence_order(&dts, &errs);
    SolverCertificate::evaluate(
        "PotentialTracerSystem",
        "domain_physics",
        3.8,
        order,
        errs[3],
        1e-7,
    )
}

fn certify_hyperbolic_billiards() -> Result<SolverCertificate, String> {
    let kind = BilliardDomainKind::RegularPolygon { p: 7, q: 3 };
    let start = DiskPoint::new(0.08, -0.05).map_err(|e| format!("{e:?}"))?;
    let mut coarse =
        TwinBilliardSystem::new(kind, 2026_1786, 1e-5).map_err(|e| format!("{e:?}"))?;
    let mut fine = TwinBilliardSystem::new(kind, 2026_1786, 1e-5).map_err(|e| format!("{e:?}"))?;
    coarse
        .set_launch_state(start, 0.43, 1e-5)
        .map_err(|e| format!("{e:?}"))?;
    fine.set_launch_state(start, 0.43, 1e-5)
        .map_err(|e| format!("{e:?}"))?;

    let (_, m_coarse) = VerifiedEngine::run_verified(|| {
        for _ in 0..10 {
            coarse.step(0.6);
        }
    });
    let (_, m_fine) = VerifiedEngine::run_verified(|| {
        for _ in 0..40 {
            fine.step(0.15);
        }
    });
    assert!(m_coarse.function_calls > 0 && m_fine.function_calls > 0);
    assert!(
        fine.total_bounces() > 0,
        "Expected at least one geodesic wall bounce"
    );

    let pos_err = (coarse.primary_ray().position.as_complex()
        - fine.primary_ray().position.as_complex())
    .norm();
    let unit_dir_err = (fine.primary_ray().direction.norm() - 1.0).abs();
    let fine_err = pos_err.max(unit_dir_err);
    Ok(SolverCertificate::evaluate(
        "TwinBilliardSystem",
        "pure_math",
        0.0,
        1.0,
        fine_err,
        1e-11,
    ))
}

fn certify_platt_scaler() -> Result<SolverCertificate, String> {
    let logits = [-3.0_f64, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0];
    let outcomes = [false, false, false, true, false, true, true, true];
    let cfg = PlattConfig {
        max_iter: 100,
        tol: 1e-11,
        l2_reg: 1e-6,
        target_smoothing: false,
    };
    let scaler =
        PlattScaler::fit_with_config(&logits, &outcomes, cfg).map_err(|e| format!("{e}"))?;
    let zero_prob = UnitInterval::new(0.0).map_err(|e| format!("{e}"))?;
    let mut probs = [zero_prob; 8];
    let (_, metrics) = VerifiedEngine::run_verified(|| {
        scaler
            .predict_into(&logits, &mut probs)
            .expect("predict_into");
    });
    assert!(metrics.function_calls > 0);

    let inv_n = 1.0 / (logits.len() as f64);
    let (mut g_a, mut g_b) = (0.0_f64, 0.0_f64);
    for (&z, &y) in logits.iter().zip(outcomes.iter()) {
        let p = 1.0 / (1.0 + (-(scaler.scale * z + scaler.bias)).exp());
        let diff = p - if y { 1.0 } else { 0.0 };
        g_a += diff * z;
        g_b += diff;
    }
    let grad_norm =
        (g_a * inv_n + cfg.l2_reg * scaler.scale).hypot(g_b * inv_n + cfg.l2_reg * scaler.bias);

    Ok(SolverCertificate::evaluate(
        "PlattScaler",
        "pure_math",
        0.0,
        2.0,
        grad_norm,
        1e-9,
    ))
}

fn certify_isotonic_pava() -> Result<SolverCertificate, String> {
    let scores = [0.1_f64, 0.2, 0.3, 0.4];
    let targets = [0.8_f64, 0.2, 0.9, 0.3];
    let mut scratch = [PavaBlock::default(); 4];
    let (block_count, m_fit) = VerifiedEngine::run_verified(|| {
        fit_pava_sorted_into(&scores, &targets, None, &mut scratch).expect("fit_pava_sorted_into")
    });
    assert!(m_fit.function_calls > 0);
    assert_eq!(block_count, 2);

    let err_b0 = (scratch[0].value - 0.5).abs();
    let err_b1 = (scratch[1].value - 0.6).abs();

    let cal = IsotonicCalibrator::fit(
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        &[false, true, false, true, true, true],
    )
    .map_err(|e| format!("{e}"))?;
    let query = [0.05_f64, 0.15, 0.25, 0.35, 0.45, 0.65];
    let zero_prob = UnitInterval::new(0.0).map_err(|e| format!("{e}"))?;
    let mut out = [zero_prob; 6];
    let (_, m_pred) = VerifiedEngine::run_verified(|| {
        cal.predict_into(&query, &mut out).expect("predict_into");
    });
    assert!(m_pred.function_calls > 0);
    let mono_violation = out
        .windows(2)
        .map(|w| (w[0].value() - w[1].value()).max(0.0))
        .fold(0.0_f64, f64::max);

    Ok(SolverCertificate::evaluate(
        "IsotonicCalibrator",
        "pure_math",
        0.0,
        1.0,
        err_b0.max(err_b1).max(mono_violation),
        1e-14,
    ))
}

/// Certifies potential-flow RK4 dye advection, exact SU(1,1) hyperbolic billiards, damped-Newton
/// Platt scaling, and zero-allocation PAVA isotonic calibration (4 solvers).
pub fn certify_calibration_and_hyperbolic() -> Result<Vec<SolverCertificate>, String> {
    Ok(vec![
        certify_potential_tracer_rk4(),
        certify_hyperbolic_billiards()?,
        certify_platt_scaler()?,
        certify_isotonic_pava()?,
    ])
}
