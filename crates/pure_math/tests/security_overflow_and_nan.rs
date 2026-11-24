//! Tier 5 security tests for `pure_math`: `checked_mul` overflow errors,
//! zero/minimum-dimension guards, `NaN`/infinity rejection, subnormal stability,
//! and zero-allocation execution inside `VerifiedEngine::run_verified`.

use math_commons::error::MathError;
use nalgebra::Point3;
use oxidize_core::rng::OxidizeRng;
use pure_math::error::{MarkovError, MatchModelError, TdaError};
use pure_math::pure_math::analysis::integration::Trapezoidal;
use pure_math::pure_math::analysis::transforms::fourier::continuous::try_calculate_fourier_coefficients_complex;
use pure_math::pure_math::differential_geometry::buckled_flow::{
    BuckledSurfaceFlow, SurfaceTopology,
};
use pure_math::pure_math::differential_geometry::surface::ParametricSurface;
use pure_math::pure_math::statistics::markov::parrondo::{
    ParrondoEnsemble, ParrondoParams, ParrondoPhaseGridConfig, ParrondoSchedule,
    scan_parrondo_phase_region,
};
use pure_math::pure_math::statistics::match_models::grid::ScoreGrid;
use pure_math::pure_math::statistics::tda::{Point3D, PointCloud3D, takens_delay_embedding_3d};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct UnitPatch;

impl ParametricSurface for UnitPatch {
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        Point3::new(u, v, 0.05 * u * v)
    }
}

#[test]
fn buckled_flow_rejects_overflow_zero_and_non_finite_inputs() {
    assert!(matches!(
        BuckledSurfaceFlow::try_from_parametric(
            &UnitPatch,
            SurfaceTopology::OpenNeumann,
            (0.0, 1.0),
            (0.0, 1.0),
            0,
            6,
        ),
        Err(MathError::OutOfBounds { .. })
    ));
    assert!(matches!(
        BuckledSurfaceFlow::try_from_parametric(
            &UnitPatch,
            SurfaceTopology::OpenNeumann,
            (0.0, 1.0),
            (0.0, 1.0),
            usize::MAX,
            4,
        ),
        Err(MathError::CapacityOverflow { .. })
    ));
    assert!(matches!(
        BuckledSurfaceFlow::try_from_parametric(
            &UnitPatch,
            SurfaceTopology::OpenNeumann,
            (f64::NAN, 1.0),
            (0.0, 1.0),
            6,
            6,
        ),
        Err(MathError::NumericalError { .. })
    ));
    assert!(matches!(
        BuckledSurfaceFlow::try_from_parametric(
            &UnitPatch,
            SurfaceTopology::OpenNeumann,
            (0.0, f64::INFINITY),
            (0.0, 1.0),
            6,
            6,
        ),
        Err(MathError::NumericalError { .. })
    ));
}

#[test]
fn fourier_rejects_overflow_zero_and_non_finite_inputs() {
    let trap = Trapezoidal::new(16);
    assert!(matches!(
        try_calculate_fourier_coefficients_complex(|x| x.cos(), 0.0, 4, &trap),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(
        try_calculate_fourier_coefficients_complex(|x| x.cos(), f64::NAN, 4, &trap),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(
        try_calculate_fourier_coefficients_complex(|x| x.cos(), f64::INFINITY, 4, &trap),
        Err(MathError::StrictlyOutOfBounds { .. })
    ));
    assert!(matches!(
        try_calculate_fourier_coefficients_complex(|x| x.cos(), 1.0, usize::MAX, &trap),
        Err(MathError::CapacityOverflow { .. })
    ));
}

#[test]
fn parrondo_rejects_overflow_zero_and_non_finite_inputs() {
    let sched = ParrondoSchedule::default();
    let mut cfg = ParrondoPhaseGridConfig {
        p_a_steps: usize::MAX,
        epsilon_steps: 2,
        ..Default::default()
    };
    assert!(matches!(
        scan_parrondo_phase_region(&cfg, &sched),
        Err(MarkovError::Math(MathError::CapacityOverflow { .. }))
    ));

    cfg.p_a_steps = 0;
    assert!(matches!(
        scan_parrondo_phase_region(&cfg, &sched),
        Err(MarkovError::InvalidState { .. })
    ));

    cfg.p_a_steps = 4;
    cfg.p_a_min = f64::NAN;
    assert!(matches!(
        scan_parrondo_phase_region(&cfg, &sched),
        Err(MarkovError::InvalidState { .. })
    ));
    assert!(ParrondoEnsemble::with_capacity(0, 10).is_err());
}

#[test]
fn tda_and_score_grid_handle_nan_subnormals_and_zero_allocation_steps() {
    assert_eq!(
        PointCloud3D::new(vec![]).err(),
        Some(TdaError::EmptyPointCloud)
    );
    assert!(PointCloud3D::new(vec![Point3D::new(f64::NAN, 0.0, 0.0)]).is_err());
    assert!(PointCloud3D::new(vec![Point3D::new(0.0, f64::INFINITY, 0.0)]).is_err());
    assert!(takens_delay_embedding_3d(&[1.0, f64::NAN, 2.0, 3.0], 1).is_err());
    assert!(takens_delay_embedding_3d(&[1.0, 2.0, 3.0], 0).is_err());

    assert!(matches!(
        ScoreGrid::from_fn(usize::MAX, true, |_, _| 0.1),
        Err(MatchModelError::InvalidMaxGoals { .. })
    ));
    assert!(matches!(
        ScoreGrid::from_fn(2, true, |_, _| f64::NAN),
        Err(MatchModelError::InvalidProbabilityMass { .. })
    ));
    assert!(matches!(
        ScoreGrid::from_fn(2, true, |_, _| -0.1),
        Err(MatchModelError::NegativeCellProbability { .. })
    ));

    let subnormal = f64::MIN_POSITIVE * 0.5;
    let params = ParrondoParams::default();
    let mut ensemble = ParrondoEnsemble::with_capacity(16, 8).expect("valid ensemble");
    let mut rng = OxidizeRng::from_seed(123);

    let ((grid_res, mean_wealth), metrics) = VerifiedEngine::run_verified(|| {
        let grid_res = ScoreGrid::from_fn(2, true, |h, a| subnormal * ((h + a + 1) as f64));
        let mean_wealth = ensemble.step_mixture(&params, &mut rng);
        (grid_res, mean_wealth)
    });

    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    let grid = grid_res.expect("subnormal positive score grid");
    assert!((grid.prob(0, 0) + grid.prob(0, 1)).is_finite());
    assert!(mean_wealth.is_finite());
}
