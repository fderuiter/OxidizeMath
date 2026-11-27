//! Integration tests verifying zero-allocation `DiagnosticVisitor` traversal, `From<LinalgError>`
//! causal `source()` retention across every crate umbrella, and `?` conversion across all
//! cross-crate methods returning `FitError`, `MonteCarloError`, `TensorError`, or `OptimizationError`.

use core::fmt;
use std::error::Error as StdError;

use oxidizemath::diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use oxidizemath::error::{ArenaError, Error, FitError, LinalgError, MathCommonsError, MathError};
use oxidizemath::math_kernel::types::Dimension;
use oxidizemath::verified_engine::allocator::VerifiedAllocator;
use oxidizemath::verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct CountingStackVisitor {
    fields: usize,
}

impl DiagnosticVisitor for CountingStackVisitor {
    fn record_str(&mut self, _key: &str, _value: &str) {
        self.fields += 1;
    }

    fn record_display(&mut self, _key: &str, _value: &dyn fmt::Display) {
        self.fields += 1;
    }
}

fn count_visited_fields<D: Diagnostic>(err: &D) -> usize {
    let mut visitor = CountingStackVisitor { fields: 0 };
    err.visit_fields(&mut visitor);
    visitor.fields
}

#[test]
fn core_only_zero_alloc_diagnostics_and_linalg_error_wiring() {
    let linalg = LinalgError::WorkspaceTooSmall {
        required: 64,
        available: 16,
    };
    let math = MathError::DimensionMismatch {
        expected: Dimension(4),
        actual: Dimension(2),
    };
    let ws = MathError::WorkspaceTooSmall {
        required: 32,
        available: 8,
    };
    let arena = ArenaError::InsufficientBytes {
        required: 64,
        available: 16,
        align: 8,
    };
    let fit = FitError::SingularMatrix;
    let commons = MathCommonsError::from(arena);
    let facade = Error::from(linalg);
    let facade_arena = Error::from(arena);

    assert!(commons.source().is_some());
    assert!(facade.source().is_some());
    assert!(facade_arena.source().is_some());

    let (total_fields, _metrics) = VerifiedEngine::run_verified(|| {
        assert_eq!(linalg.severity(), Severity::Error);
        assert_eq!(arena.severity(), Severity::Error);
        count_visited_fields(&linalg)
            + count_visited_fields(&arena)
            + count_visited_fields(&math)
            + count_visited_fields(&ws)
            + count_visited_fields(&fit)
            + count_visited_fields(&commons)
            + count_visited_fields(&facade)
            + count_visited_fields(&facade_arena)
    });
    assert!(total_fields >= 22);
}

#[cfg(all(
    feature = "pure_math",
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics"
))]
mod full_domain_wiring {
    use super::*;
    use oxidizemath::error::{
        AIError, AppliedError, BatteryError, BiologyError, BraidError, ClimateError,
        EpidemiologyDomainError, GameTheoryError, HighEnergyError, KnotError, LinearAlgebraError,
        PhysicsError, PureMathError, QuantumError, SandpileError, Su11HyperbolicError,
    };

    #[test]
    fn all_umbrellas_implement_from_linalg_error_and_retain_source() {
        let raw = LinalgError::SingularMatrix;
        let arena_raw = ArenaError::ArithmeticOverflow;
        let la = LinearAlgebraError::from(raw);
        let commons = MathCommonsError::from(arena_raw);
        let pm = PureMathError::from(raw);
        let ai = AIError::from(raw);
        let applied = AppliedError::from(raw);
        let bio = BiologyError::from(raw);
        let climate = ClimateError::from(raw);
        let epi = EpidemiologyDomainError::from(raw);
        let phys = PhysicsError::from(raw);
        let facade = Error::from(raw);
        let facade_arena = Error::from(arena_raw);

        for has_src in [
            la.source().is_some(),
            commons.source().is_some(),
            pm.source().is_some(),
            ai.source().is_some(),
            applied.source().is_some(),
            bio.source().is_some(),
            climate.source().is_some(),
            epi.source().is_some(),
            phys.source().is_some(),
            facade.source().is_some(),
            facade_arena.source().is_some(),
        ] {
            assert!(has_src);
        }

        let (visited, _metrics) = VerifiedEngine::run_verified(|| {
            let su11 = Su11HyperbolicError::NonHyperbolicSchlafli { p: 4, q: 4 };
            let braid = BraidError::InvalidStrandCount { strands: 0 };
            let knot = KnotError::EmptyCurve;
            count_visited_fields(&la)
                + count_visited_fields(&pm)
                + count_visited_fields(&ai)
                + count_visited_fields(&applied)
                + count_visited_fields(&bio)
                + count_visited_fields(&climate)
                + count_visited_fields(&epi)
                + count_visited_fields(&phys)
                + count_visited_fields(&su11)
                + count_visited_fields(&braid)
                + count_visited_fields(&knot)
        });
        assert!(visited >= 30);
    }

    struct FailingOptimizer;

    impl<T: oxidizemath::nalgebra::RealField + Copy>
        oxidizemath::pure_math::analysis::optimization::Optimizer<
            T,
            (
                usize,
                oxidizemath::pure_math::analysis::optimization::ParamType,
            ),
        > for FailingOptimizer
    {
        fn update_matrix(
            &mut self,
            _key: (
                usize,
                oxidizemath::pure_math::analysis::optimization::ParamType,
            ),
            _param: &mut oxidizemath::nalgebra::DMatrix<T>,
            _grad: &oxidizemath::nalgebra::DMatrix<T>,
        ) -> Result<(), oxidizemath::error::OptimizationError> {
            Err(oxidizemath::error::OptimizationError::ConversionError)
        }

        fn update_vector(
            &mut self,
            _key: (
                usize,
                oxidizemath::pure_math::analysis::optimization::ParamType,
            ),
            _param: &mut oxidizemath::nalgebra::DVector<T>,
            _grad: &oxidizemath::nalgebra::DVector<T>,
        ) -> Result<(), oxidizemath::error::OptimizationError> {
            Err(oxidizemath::error::OptimizationError::ConversionError)
        }
    }

    #[test]
    fn ai_optimization_methods_convert_into_ai_error_via_question_mark() {
        use oxidizemath::ai::deep_learning_theory::cycle::TrainingLoop;
        use oxidizemath::ai::deep_learning_theory::model::{Trainable, TwoLayerMLP};
        use oxidizemath::nalgebra::DVector;

        let run_mlp = || -> Result<(), AIError> {
            let mut mlp = TwoLayerMLP::new(2, 3, 2);
            let mut opt = FailingOptimizer;
            let x = DVector::from_vec(vec![1.0, 0.5]);
            let grad = DVector::from_vec(vec![0.1, -0.1]);
            mlp.backward_update(&x, &grad, &mut opt)?;
            Ok(())
        };
        let run_loop = || -> Result<f64, AIError> {
            let mut trainer = TrainingLoop::new(2, 3, 2, Box::new(FailingOptimizer));
            let x = DVector::from_vec(vec![1.0, 0.5]);
            let y = DVector::from_vec(vec![1.0, 0.0]);
            Ok(trainer.train_step(&x, &y)?)
        };
        assert!(run_mlp().unwrap_err().source().is_some());
        assert!(run_loop().unwrap_err().source().is_some());
    }

    #[test]
    fn climate_optimization_methods_convert_into_climate_error_via_question_mark() {
        use oxidizemath::climate::autoencoder::{Autoencoder, ConvLayer};
        use oxidizemath::climate::predictor::Predictor;
        use oxidizemath::nalgebra::{DMatrix, DVector};
        use oxidizemath::pure_math::analysis::optimization::Trainable;

        let check_conv = || -> Result<DMatrix<f32>, ClimateError> {
            let mut conv = ConvLayer::new(2, 2);
            let mut opt = FailingOptimizer;
            let x = DMatrix::from_element(1, 2, 1.0_f32);
            let g = DMatrix::from_element(1, 2, 0.5_f32);
            Ok(conv.backward_update(&x, &g, &mut opt, 0)?)
        };
        let check_ae = || -> Result<(), ClimateError> {
            let mut ae = Autoencoder::new(2, 3);
            let mut opt = FailingOptimizer;
            let x = DVector::from_element(2, 1.0_f32);
            let g = DVector::from_element(2, 0.5_f32);
            ae.backward_update(&x, &g, &mut opt)?;
            Ok(())
        };
        let check_pred = || -> Result<(), ClimateError> {
            let mut pred = Predictor::new(4, 2);
            let mut opt = FailingOptimizer;
            let x = DVector::from_element(4, 1.0_f32);
            let g = DVector::from_element(2, 0.5_f32);
            pred.backward_update(&x, &g, &mut opt)?;
            Ok(())
        };
        assert!(check_conv().unwrap_err().source().is_some());
        assert!(check_ae().unwrap_err().source().is_some());
        assert!(check_pred().unwrap_err().source().is_some());
    }

    #[test]
    fn applied_battery_and_mechanism_methods_convert_via_question_mark() {
        use oxidizemath::applied::battery_degradation::{
            ChemistryPreset, DepthOfDischarge, ElectrochemicalAgingModel,
            calibrate_electrochemical_model,
        };
        use oxidizemath::applied::game_theory::mechanism_design::{
            distribution::Uniform, simulate_optimal_revenue_stats,
            simulate_optimal_revenue_stats_with_rng,
        };
        use oxidizemath::rng::OxidizeRng;

        let model = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Nmc811);
        let dod = DepthOfDischarge::new_clamped(0.8);
        let too_few = [[10.0_f64, 0.99_f64]];

        let b1 = (|| -> Result<_, BatteryError> {
            Ok(calibrate_electrochemical_model(&model, &too_few, dod)?)
        })()
        .unwrap_err();
        let b2 =
            (|| -> Result<_, AppliedError> { Ok(model.calibrate_from_cycles(&too_few, dod)?) })()
                .unwrap_err();
        assert!(b1.source().is_some() && b2.source().is_some());

        let dist = Uniform::new(0.0, 1.0).unwrap();
        let g1 = (|| -> Result<_, GameTheoryError> {
            Ok(simulate_optimal_revenue_stats(&dist, 2, 0)?)
        })()
        .unwrap_err();
        let g2 = (|| -> Result<_, AppliedError> {
            let mut rng = OxidizeRng::new(7);
            Ok(simulate_optimal_revenue_stats_with_rng(
                &dist, 2, 0, 0.95, &mut rng,
            )?)
        })()
        .unwrap_err();
        assert!(g1.source().is_some() && g2.source().is_some());
    }

    #[test]
    fn applied_graph_replicator_system_converts_and_visits_zero_alloc() {
        use oxidizemath::applied::game_theory::evolutionary::GraphReplicatorSystem;
        use oxidizemath::nalgebra::DMatrix;

        let non_square = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let square = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 4.0, 1.0]);

        let app_err = (|| -> Result<GraphReplicatorSystem, AppliedError> {
            Ok(GraphReplicatorSystem::new(
                2,
                &non_square,
                &[(0, 1, 1.0)],
                &[0.5, 0.5],
            )?)
        })()
        .unwrap_err();
        let facade_err = (|| -> Result<GraphReplicatorSystem, Error> {
            Ok(GraphReplicatorSystem::from_population_mean(
                0,
                &square,
                &[],
                &[0.5, 0.5],
            )?)
        })()
        .unwrap_err();
        assert!(app_err.source().is_some() && facade_err.source().is_some());

        let (fields, _metrics) = VerifiedEngine::run_verified(|| {
            count_visited_fields(&app_err) + count_visited_fields(&facade_err)
        });
        assert!(fields >= 6);
    }

    #[test]
    fn physics_sandpile_methods_convert_via_question_mark() {
        use oxidizemath::physics::stat_mech::sandpile::{
            AvalancheObservable, AvalancheTelemetry, SandpileGrid,
        };

        let grid = SandpileGrid::new(4, 4, Some(1)).unwrap();
        let s1 = (|| -> Result<_, SandpileError> {
            Ok(grid.fit_avalanche_power_law(AvalancheObservable::Size)?)
        })()
        .unwrap_err();
        let s2 = (|| -> Result<_, PhysicsError> {
            let telemetry = AvalancheTelemetry::default();
            Ok(telemetry.fit_power_law(AvalancheObservable::Area, 16)?)
        })()
        .unwrap_err();
        assert!(s1.source().is_some() && s2.source().is_some());
    }

    #[test]
    fn physics_general_relativity_static_metric_methods_convert_via_question_mark() {
        use oxidizemath::nalgebra::SVector;
        use oxidizemath::physics::high_energy::general_relativity::{
            KerrEquatorialMetric, MorrisThorneWormhole, SchwarzschildBlackHole,
        };
        use oxidizemath::pure_math::tensor::StaticMetric;

        let bh = SchwarzschildBlackHole::from_schwarzschild_radius(2.0).unwrap();
        let bad4 = SVector::<f64, 4>::new(0.0, 1.0, 1.5, 0.0);
        let bad3 = SVector::<f64, 3>::new(f64::NAN, 0.0, 0.0);
        let wh = MorrisThorneWormhole::new(1.0).unwrap();
        let kerr = KerrEquatorialMetric::new(2.0, 0.4).unwrap();

        let errs: [HighEnergyError; 9] = [
            (|| -> Result<_, HighEnergyError> {
                bh.metric_at(&bad4)?;
                Ok(())
            })()
            .unwrap_err(),
            (|| -> Result<_, HighEnergyError> {
                bh.inverse_metric_at(&bad4)?;
                Ok(())
            })()
            .unwrap_err(),
            (|| -> Result<_, HighEnergyError> {
                bh.christoffel_at(&bad4)?;
                Ok(())
            })()
            .unwrap_err(),
            (|| -> Result<_, HighEnergyError> {
                wh.metric_at(&bad3)?;
                Ok(())
            })()
            .unwrap_err(),
            (|| -> Result<_, HighEnergyError> {
                wh.inverse_metric_at(&bad3)?;
                Ok(())
            })()
            .unwrap_err(),
            (|| -> Result<_, HighEnergyError> {
                wh.christoffel_at(&bad3)?;
                Ok(())
            })()
            .unwrap_err(),
            (|| -> Result<_, HighEnergyError> {
                kerr.metric_at(&bad4)?;
                Ok(())
            })()
            .unwrap_err(),
            (|| -> Result<_, HighEnergyError> {
                kerr.inverse_metric_at(&bad4)?;
                Ok(())
            })()
            .unwrap_err(),
            (|| -> Result<_, HighEnergyError> {
                kerr.christoffel_at(&bad4)?;
                Ok(())
            })()
            .unwrap_err(),
        ];
        assert!(errs.iter().all(|e| e.source().is_some()));
        let phys_err: PhysicsError = errs[0].clone().into();
        assert!(phys_err.source().is_some());
    }

    #[test]
    fn quantum_error_converts_into_physics_and_facade_and_visits_zero_alloc() {
        use oxidizemath::physics::quantum::OrbitalQuantumNumbers;

        let (fields, _metrics) = VerifiedEngine::run_verified(|| {
            let q_err: QuantumError = OrbitalQuantumNumbers::new(0, 0, 0).unwrap_err();
            let phys_err = (|| -> Result<OrbitalQuantumNumbers, PhysicsError> {
                Ok(OrbitalQuantumNumbers::new(3, 3, 0)?)
            })()
            .unwrap_err();
            let facade_err = (|| -> Result<OrbitalQuantumNumbers, Error> {
                Ok(OrbitalQuantumNumbers::new(2, 1, -2)?)
            })()
            .unwrap_err();
            assert!(phys_err.source().is_some() && facade_err.source().is_some());
            count_visited_fields(&q_err)
                + count_visited_fields(&phys_err)
                + count_visited_fields(&facade_err)
        });
        assert!(fields >= 14);
    }
}
