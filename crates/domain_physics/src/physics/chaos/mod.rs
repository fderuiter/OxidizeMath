//! # Deterministic Chaos
//!
//! Chaos theory studies the behavior of dynamical systems that are highly sensitive to initial conditions.
//! This sensitivity, often referred to as the **Butterfly Effect**, implies that small differences in initial
//! states yield widely diverging outcomes, rendering long-term prediction impossible.
//!
//! ## The Concept: Sensitivity to Initial Conditions
//!
//! Even in a deterministic system (no randomness), two trajectories starting arbitrarily close together
//! can diverge exponentially over time.
//!
//! ```mermaid
//! graph LR
//!     Start((Start)) -->|t=0| S1[State A]
//!     Start -->|t=0 + ε| S2[State A + ε]
//!
//!     subgraph "Time Evolution"
//!     S1 -->|t=10| M1[State B]
//!     S2 -->|t=10| M2[State B']
//!
//!     M1 -->|t=100| E1[State C]
//!     M2 -->|t=100| E2[State Z]
//!     end
//!
//!     style E1 fill:#aaf,stroke:#333
//!     style E2 fill:#faa,stroke:#333
//! ```
//!
//! ## Modules
//!
//! *   **[Billiards](billiards)**: Exact flat Euclidean ray billiards (circular, elliptical, Bunimovich stadium, Sinai) and Birkhoff maps.
//! *   **[Double Pendulum](double_pendulum)**: Canonical non-separable Hamiltonian double pendulum.
//! *   **[Fractals](fractals)**: Geometric structures with fractional dimension (e.g., Cantor Set).
//! *   **[Hamiltonian Chaos](hamiltonian)**: Unified zero-allocation conservative flows, Chirikov standard map, and symplectic ensemble helpers.
//! *   **[Hénon-Heiles](henon_heiles)**: Separable 2-DOF galactic Hamiltonian with KAM islands and chaotic seas.
//! *   **[Logistic Map](logistic)**: Discrete-time demographic model $x_{n+1} = r x_n (1 - x_n)$.
//! *   **[Lorenz System](lorenz)**: Continuous-time atmospheric convection model (The "Butterfly" attractor).
//! *   **[Metrics](metrics)**: Tools to quantify chaos (Lyapunov Exponents, Correlation Dimension).
//! *   **[Poincaré Sections](poincare)**: Zero-allocation cubic-Hermite surface-of-section detector.
//!
//! ##  Deep Dive: The Lorenz Attractor
//!
//! The Lorenz system is defined by three coupled differential equations:
//!
//! $$ \frac{dx}{dt} = \sigma(y - x), \quad \frac{dy}{dt} = x(\rho - z) - y, \quad \frac{dz}{dt} = xy - \beta z $$
//!
//! ### Example Implementation
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
//!
//! // 1. Initialize the system with the "Butterfly" parameters
//! // sigma=10, rho=28, beta=8/3
//! let initial_state = LorenzState::new(1.0, 1.0, 1.0);
//! let mut system = LorenzBuilder::new()
//!     .sigma(10.0)
//!     .rho(28.0)
//!     .beta(8.0 / 3.0)
//!     .build(initial_state);
//!
//! // 2. Simulate forward in time
//! let dt = 0.01;
//! for _ in 0..1000 {
//!     system.step(dt);
//! }
//!
//! // 3. Observe the result (The state stays bounded within the attractor)
//! println!("Final State: {:.4}, {:.4}, {:.4}",
//!     system.state.vec.x,
//!     system.state.vec.y,
//!     system.state.vec.z
//! );
//! assert!(system.state.vec.norm() < 100.0); // Simple boundedness check
//! ```

pub mod billiards;
pub mod double_pendulum;
pub mod fractals;
pub mod hamiltonian;
pub mod henon_heiles;
pub mod logistic;
pub mod lorenz;
pub mod metrics;
pub mod poincare;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra as na;
    use scientific_metadata::theory::TheoryDescribable;

    #[test]
    #[verified_engine::verified]
    fn test_logistic_chaos_lyapunov() {
        // For r=3.9, the map is chaotic, so Lyapunov exponent should be positive.
        // Theoretical value is approx 0.496.
        // We use x0=0.1 to avoid starting at the critical point x=0.5.
        let lambda = metrics::logistic_lyapunov(3.9, 0.1, 1000);
        assert!(
            lambda > 0.0,
            "Lyapunov exponent for r=3.9 should be positive, got {}",
            lambda
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_logistic_stability_lyapunov() {
        // For r=2.5, the map is stable (fixed point at 0.6), so Lyapunov exponent should be negative.
        // f'(x*) = 2.5(1 - 1.2) = 2.5(-0.2) = -0.5. ln(0.5) approx -0.693.
        // We use x0=0.1.
        let lambda = metrics::logistic_lyapunov(2.5, 0.1, 1000);
        assert!(
            lambda < 0.0,
            "Lyapunov exponent for r=2.5 should be negative, got {}",
            lambda
        );
        // We can be more precise
        assert!(
            (lambda - -std::f64::consts::LN_2).abs() < 0.1,
            "Expected approx -0.693, got {}",
            lambda
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_lorenz_boundedness() {
        // Run Lorenz system and ensure it stays within reasonable bounds.
        // The Lorenz attractor is contained within a specific region of space.
        let state = lorenz::LorenzState::new(1.0, 1.0, 1.0);
        let mut system = lorenz::LorenzSystem::default_chaotic(state);

        let dt = 0.01;
        for _ in 0..1000 {
            system.step(dt);
            let s = system.state.vec;
            assert!(s.x.abs() < 100.0, "x diverged: {}", s.x);
            assert!(s.y.abs() < 100.0, "y diverged: {}", s.y);
            assert!(s.z.abs() < 100.0, "z diverged: {}", s.z);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_bifurcation_diagram_generation() {
        let points = logistic::generate_bifurcation_diagram(3.0, 4.0, 10);
        // steps=10 means 11 values of r. Each has 50 points. Total 550 points.
        assert_eq!(points.len(), 11 * 50);
        // Check bounds
        for (r, x) in points {
            assert!((3.0..=4.0).contains(&r));
            assert!((0.0..=1.0).contains(&x));
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_lorenz_lyapunov_strategy() {
        use pure_math::pure_math::analysis::ode::{Euler, RungeKutta4};
        let state = lorenz::LorenzState::new(1.0, 1.0, 1.0);
        let system = lorenz::LorenzSystem::default_chaotic(state);

        // Test with RK4
        let dummy_state = na::Vector3::zeros();
        let lambda_rk4 = metrics::lorenz_lyapunov(
            &system,
            &mut RungeKutta4::new(&dummy_state),
            na::Vector3::new(10.0, 10.0, 10.0),
            0.01,
            100,
            1.0,
        )
        .unwrap();

        assert!(
            lambda_rk4 > 0.0,
            "Lorenz with RK4 should be chaotic, got {}",
            lambda_rk4
        );

        // Test with Euler (less accurate, but should run)
        let lambda_euler = metrics::lorenz_lyapunov(
            &system,
            &mut Euler::new(&dummy_state),
            na::Vector3::new(10.0, 10.0, 10.0),
            0.0001, // Euler needs smaller step for stability
            100,
            1.0,
        )
        .unwrap();

        assert!(
            lambda_euler > 0.0,
            "Lorenz with Euler should be chaotic, got {}",
            lambda_euler
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_dimension_simple() {
        // Create a line of points: (0,0,0), (1,0,0), (2,0,0) ...
        // Dimension should be 1. But we just test the C(epsilon) calculation here.
        let mut traj = Vec::new();
        for i in 0..10 {
            traj.push(na::Vector3::new(i as f64, 0.0, 0.0));
        }

        // Epsilon = 1.1. Pairs with dist < 1.1 are adjacent points.
        // Pairs (0,1), (1,2), ..., (8,9). There are 9 such pairs.
        // Total pairs N(N-1) = 10*9 = 90.
        // Count = 9 * 2 = 18. (symmetric)
        // C = 18 / 90 = 0.2.

        let c = fractals::correlation_dimension(&traj, 1.1);
        assert_relative_eq!(c, 0.2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_lorenz_builder_custom_parameters() {
        let state = lorenz::LorenzState::new(1.0, 1.0, 1.0);

        // Build a system with non-standard parameters
        // Example: sigma = 11.0, rho = 29.0 (slightly different from standard)
        let system = lorenz::LorenzBuilder::new()
            .sigma(11.0)
            .rho(29.0)
            .beta(3.0)
            .build(state);

        assert_eq!(system.sigma, 11.0);
        assert_eq!(system.rho, 29.0);
        assert_eq!(system.beta, 3.0);

        // Ensure it runs
        let mut running_system = system;
        running_system.step(0.01);
        assert!(running_system.state.vec.x.is_finite());
    }

    #[test]
    #[verified_engine::verified]
    fn test_logistic_lyapunov_zero_steps_and_superstable() {
        assert_eq!(metrics::logistic_lyapunov(3.8, 0.4, 0), 0.0);
        let superstable = metrics::logistic_lyapunov(2.0, 0.5, 64);
        assert!(superstable.is_finite() && superstable < -20.0);
        let escaping = metrics::logistic_lyapunov(4.5, 2.0, 100);
        assert!(escaping.is_finite() && escaping > 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_lorenz_lyapunov_zero_distance_contraction_zero_alloc() {
        use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4};

        struct InstantCollapseSink;
        impl OdeSystem<na::Vector3<f64>> for InstantCollapseSink {
            fn derivative(&self, _t: f64, state: &na::Vector3<f64>) -> na::Vector3<f64> {
                -1000.0 * (*state)
            }
        }

        let init = na::Vector3::zeros();
        let mut rk4 = RungeKutta4::new(&init);
        let lambda =
            metrics::lorenz_lyapunov(&InstantCollapseSink, &mut rk4, init, 0.002, 5, 0.1).unwrap();
        assert!(lambda.is_finite() && lambda < 0.0);

        let err = metrics::lorenz_lyapunov(&InstantCollapseSink, &mut rk4, init, 0.5, 5, 0.1);
        assert!(err.is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_chaos_kernels_zero_allocations_under_verified_allocator() {
        use pure_math::pure_math::analysis::ode::RungeKutta4;
        use verified_engine::engine::VerifiedEngine;

        let chaotic =
            lorenz::LorenzSystem::default_chaotic(lorenz::LorenzState::new(1.0, 1.0, 1.0));
        let stable = lorenz::LorenzBuilder::new()
            .sigma(10.0)
            .rho(0.5)
            .beta(8.0 / 3.0)
            .build(lorenz::LorenzState::new(1.0, 1.0, 1.0));
        let decoupled = lorenz::LorenzBuilder::new()
            .sigma(0.0)
            .rho(28.0)
            .beta(8.0 / 3.0)
            .build(lorenz::LorenzState::new(1.0, 1.0, 1.0));
        let mut solver = RungeKutta4::new(&na::Vector3::zeros());
        let mut scratch = [
            na::Vector3::new(9.0, 0.0, 0.0),
            na::Vector3::new(3.0, 0.0, 0.0),
            na::Vector3::new(0.0, 0.0, 0.0),
            na::Vector3::new(1.0, 0.0, 0.0),
            na::Vector3::new(2.0, 0.0, 0.0),
            na::Vector3::new(4.0, 0.0, 0.0),
            na::Vector3::new(8.0, 0.0, 0.0),
            na::Vector3::new(7.0, 0.0, 0.0),
            na::Vector3::new(6.0, 0.0, 0.0),
            na::Vector3::new(5.0, 0.0, 0.0),
        ];

        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((lam_zero, lam_sing, lam_chaos, lam_conv, lam_sig0, corr), m) =
            VerifiedEngine::run_verified(|| {
                let lz = metrics::logistic_lyapunov(3.9, 0.1, 0);
                let ls = metrics::logistic_lyapunov(2.0, 0.5, 250);
                let lc = metrics::lorenz_lyapunov(
                    &chaotic,
                    &mut solver,
                    na::Vector3::new(10.0, 10.0, 10.0),
                    0.01,
                    24,
                    0.25,
                )
                .unwrap();
                let lconv = metrics::lorenz_lyapunov(
                    &stable,
                    &mut solver,
                    na::Vector3::new(0.0, 0.0, 0.0),
                    0.02,
                    30,
                    1.0,
                )
                .unwrap();
                let lsig0 = metrics::lorenz_lyapunov(
                    &decoupled,
                    &mut solver,
                    na::Vector3::new(0.0, 1.0, 1.0),
                    0.05,
                    100,
                    2.0,
                )
                .unwrap();
                let c_in_place = fractals::correlation_dimension_in_place(&mut scratch, 1.1);
                (lz, ls, lc, lconv, lsig0, c_in_place)
            });

        assert_eq!(lam_zero, 0.0);
        assert!(lam_sing.is_finite() && lam_sing < 0.0);
        assert!(lam_chaos.is_finite() && lam_chaos > 0.0);
        assert!(lam_conv.is_finite() && lam_conv < 0.0);
        assert!(lam_sig0.is_finite() && lam_sig0 <= 0.02);
        assert_relative_eq!(corr, 0.2);
        assert!(m.function_calls > 0);
    }

    #[test]
    fn test_lorenz_theory_description_classifies_four_regimes() {
        let s0 = lorenz::LorenzState::new(1.0, 1.0, 1.0);
        let origin = lorenz::LorenzBuilder::new().rho(0.5).build(s0);
        assert!(origin.theory_description().contains("Origin sink regime"));

        let twin = lorenz::LorenzBuilder::new().rho(15.0).build(s0);
        assert!(
            twin.theory_description()
                .contains("Twin-lobe equilibrium regime")
        );

        let sigma_zero = lorenz::LorenzBuilder::new().sigma(0.0).rho(28.0).build(s0);
        assert!(
            !sigma_zero.theory_description().contains("Chaotic regime"),
            "sigma = 0.0 with rho = 28.0 must not classify as Chaotic regime, got: {}",
            sigma_zero.theory_description()
        );

        let chaotic = lorenz::LorenzBuilder::new().rho(28.0).build(s0);
        assert!(chaotic.theory_description().contains("Chaotic regime"));

        let periodic = lorenz::LorenzBuilder::new().rho(350.0).build(s0);
        assert!(
            periodic
                .theory_description()
                .contains("Periodic window regime")
        );
    }
}

// [cite:chaos_theory]

use pure_math::theory_verification;

theory_verification!(
    module = "chaos",
    constants = {
        HENON_HEILES_ESCAPE_ENERGY = 1.0 / 6.0;
        CHIRIKOV_CRITICAL_K = 0.971_635_406;
        UNIT_CIRCLE_PERIMETER = core::f64::consts::TAU;
    },
    test = {
        use approx::assert_relative_eq;
        let hh = henon_heiles::HenonHeiles::default();
        assert_relative_eq!(
            hh.escape_energy(),
            HENON_HEILES_ESCAPE_ENERGY,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            hamiltonian::CRITICAL_K_GREENE,
            CHIRIKOV_CRITICAL_K,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        let disk = billiards::CircularBilliard::new(1.0);
        assert_relative_eq!(
            disk.perimeter(),
            UNIT_CIRCLE_PERIMETER,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
