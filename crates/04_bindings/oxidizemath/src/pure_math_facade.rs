//! Unified facade module preserving all `oxidizemath::pure_math` and `math_explorer::pure_math`
//! paths while composing Tier 2 engine solvers (`engine_linalg`, `engine_pde`) at Tier 4.

/// Abstract algebra structures paired with `nalgebra`-backed numerical linear algebra solvers.
pub mod algebra {
    pub use ::pure_math::pure_math::algebra::{
        AbelianGroup, BraidError, BraidWord, CyclicElement, EuclideanDomain, Field, Fp, Group,
        LaurentError, LaurentPolynomial, LinearAlgebraError, MAX_BRAID_STRANDS, MAX_BRAID_WORD_LEN,
        MAX_LAURENT_SPAN, Monoid, Permutation, Polynomial, RefOps, Ring, Scalar, Semigroup, Zn,
        fields, generate_left_coset, generate_right_coset, group, is_normal_subgroup, is_subgroup,
        polynomial, traits,
    };

    /// Symbolic canonical forms, eigensolvers, and numerical linear algebra decompositions/solvers.
    pub mod linear_algebra {
        pub use ::pure_math::pure_math::algebra::linear_algebra::{
            LinearAlgebraError, canonical, eigen,
        };
        pub use engine_linalg::nalgebra_ops::{
            decomposition, effective_rank, effective_rank_with_workspace,
            linalg_solve_workspace_bytes, low_rank_approximation, moore_penrose_pseudoinverse,
            numerical, solve_cholesky_system, solve_cholesky_system_with_workspace,
            solve_conjugate_gradient, solve_conjugate_gradient_with_workspace,
            solve_gmres_restarted, solve_gmres_restarted_with_workspace, solve_linear_system,
            solve_linear_system_with_workspace, solve_normal_equation,
            solve_normal_equation_with_workspace, solve_tikhonov_system,
            solve_tikhonov_system_with_workspace, svd_decomposition_workspace_bytes,
            try_low_rank_approximation, try_low_rank_approximation_with_workspace,
            try_moore_penrose_pseudoinverse, try_moore_penrose_pseudoinverse_with_workspace,
        };
    }
}

/// Real/complex analysis, integration, root-finding, transforms, ODEs, PDEs, and Kalman/evolutionary systems.
pub mod analysis {
    pub use ::pure_math::pure_math::analysis::{
        AnalysisError, EvolutionError, FitnessStrategy, GameTheoryError, MatrixPayoff,
        OptimizationError, ReplicatorDynamics, StochasticError, TransformError, complex, evolution,
        evolutionary, integration, momentum_dynamics, ode, optimization, roots, stochastic,
        transforms,
    };
    pub use engine_diffeq::{
        KalmanError, kalman,
        kalman::{KalmanFilter, KalmanFilterBuilder, KalmanModel, KalmanSystem},
    };

    /// Partial differential equation classification, analytical/grid solvers, and stencil engines.
    pub mod pde {
        pub use ::pure_math::pure_math::analysis::pde::{
            BoundaryCondition, DirichletBoundary, NeumannBoundary, PdeClassification,
            SecondOrderLinearPde2D, greens, heat, laplace, tridiagonal, wave, wave_2d, wave_fdtd,
        };
        pub use engine_pde::advection_2d;

        /// Re-export of `#![no_std]` stencil operators and fused grid steppers from `engine_pde`.
        pub mod fused_stepper {
            pub use engine_pde::*;
        }
    }
}

pub use ::pure_math::pure_math::{
    algorithmic_information, differential_geometry, elliptic_curves, error, game_theory,
    graph_theory, hyperbolic, knot_theory, logic, number_theory, special_functions, tensor,
    vector_calculus,
};

/// Statistical modeling, competitive ranking, decision theory, and Almgren-Chriss market execution.
pub mod statistics {
    pub use ::pure_math::pure_math::statistics::{
        black_scholes, bootstrap, calibration, copula, elo, glicko2, kelly, markov, match_models,
        monte_carlo, ou_process, poisson, qmc, queueing, regression, risk, ruin, tda, validation,
        zip_regression,
    };

    /// Zero-allocation bookmaker overround removal, `+EV` valuation, Dutching, hedging, and Almgren-Chriss execution.
    pub mod market_pricing {
        pub use ::pure_math::pure_math::statistics::market_pricing::{
            AllMethodsEvReport, ArbitrageReport, DevigMetadata, DevigMethod, DutchingMode,
            DutchingReport, EvEvaluation, HedgeMode, HedgeReport, MAX_MARKET_OUTCOMES,
            MarketOverround, MarketPricingError, MiddleReport, SimplexViolationDiagnostic, booksum,
            compare_outcome_ev_all_methods, devig, devig_additive_detailed_into,
            devig_additive_into, devig_into, devig_multiplicative_into, devig_power_into,
            devig_shin_into, dutch_outcomes_into, dutch_subset_into, dutching, ev,
            evaluate_arbitrage_into, evaluate_ev_from_probability, evaluate_hedge, evaluate_middle,
            evaluate_middle_custom_stakes, evaluate_outcome_ev, evaluate_two_way_arbitrage,
            fair_decimal_odds_into, fair_odds_from_probabilities_into, implied_probabilities_into,
            is_arbitrage, market_overround, overround, turnover_margin,
        };
        pub use engine_opt::almgren_chriss::{
            AlmgrenChrissConfig, almgren_chriss_analytical_trajectory_into,
            almgren_chriss_workspace_len, solve_almgren_chriss_mpc_into,
        };

        /// Spread/total middles, live/futures hedging, and multi-period Almgren-Chriss optimal execution.
        pub mod execution {
            pub use ::pure_math::pure_math::statistics::market_pricing::execution::*;
            pub use engine_opt::almgren_chriss::{
                AlmgrenChrissConfig, almgren_chriss_analytical_trajectory_into,
                almgren_chriss_workspace_len, solve_almgren_chriss_mpc_into,
            };
        }
    }
}
