//! Cross-platform deterministic tests for `pure_math` stochastic solvers with pinned SHA-256 digests.

use nalgebra::DMatrix;
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};
use pure_math::pure_math::analysis::stochastic::{
    GillespieSolver, StochasticError, StochasticSystem,
};
use pure_math::pure_math::statistics::markov::ctmc::ContinuousMarkovChain;
use pure_math::pure_math::statistics::ou_process::{EulerMaruyama, OuParams, TimeStep};

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(not(target_arch = "wasm32"))]
static ALLOC_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const PINNED_GILLESPIE_DIGEST: &str =
    "83a0453e678338e743a1246c303e0a34f2dbfb9a2edfc14a4ad02523a573f462";
const PINNED_OU_DIGEST: &str = "ac1d0469a63e5be25f1354906c4fd20769f99063e07b08bf6c2ad241fcc38da2";
const PINNED_CTMC_DIGEST: &str = "27c1a86edaed7b94c3c4d6f027321547afe9165b4c26c1c205a810931047f8f2";

struct BirthDeathDimers {
    birth: f64,
    dimer: f64,
}

impl StochasticSystem<[i64; 2]> for BirthDeathDimers {
    #[verified_engine::verified]
    fn propensities(&self, state: &[i64; 2], out: &mut Vec<f64>) {
        let a = state[0].max(0) as f64;
        let b = state[1].max(0) as f64;
        out.push(self.birth);
        out.push(self.dimer * a * (a - 1.0).max(0.0) * 0.5);
        out.push(0.2 * b);
    }

    #[verified_engine::verified]
    fn react(&self, state: &mut [i64; 2], reaction_index: usize) -> Result<(), StochasticError> {
        match reaction_index {
            0 => state[0] += 1,
            1 => {
                state[0] -= 2;
                state[1] += 1;
            }
            2 => state[1] -= 1,
            _ => return Err(StochasticError::InvalidReactionIndex(reaction_index)),
        }
        Ok(())
    }
}

#[test]
#[verified_engine::verified]
fn test_gillespie_zero_alloc_and_pinned_digest() {
    #[cfg(not(target_arch = "wasm32"))]
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let sys = BirthDeathDimers {
        birth: 3.5,
        dimer: 0.08,
    };
    let run = |seed: u64| -> (Vec<(f64, [i64; 2])>, String) {
        let mut solver = GillespieSolver::with_capacity(OxidizeRng::new(seed), 4);
        let mut state = [18_i64, 4_i64];
        let mut trace = Vec::with_capacity(40);
        let mut digest = TrajectoryDigest::new();
        for step_idx in 0..40 {
            #[cfg(not(target_arch = "wasm32"))]
            let dt = if step_idx > 0 {
                let (dt_val, metrics) =
                    verified_engine::engine::VerifiedEngine::run_verified(|| {
                        solver.step(&sys, &mut state).unwrap()
                    });
                assert!(metrics.function_calls >= 1);
                dt_val
            } else {
                solver.step(&sys, &mut state).unwrap()
            };
            #[cfg(target_arch = "wasm32")]
            let dt = {
                let _ = step_idx;
                solver.step(&sys, &mut state).unwrap()
            };
            digest.update_f64(dt);
            digest.update_i64(state[0]);
            digest.update_i64(state[1]);
            trace.push((dt, state));
        }
        (trace, digest.finish_hex())
    };

    let (t1, h1) = run(2026);
    let (t2, h2) = run(2026);
    let (t3, h3) = run(2027);
    assert_eq!(t1, t2);
    assert_eq!(h1, h2);
    assert_ne!(t1, t3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_GILLESPIE_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_ou_euler_maruyama_pinned_digest() {
    #[cfg(not(target_arch = "wasm32"))]
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let params = OuParams::from_values(0.5, 1.2, 0.3).unwrap();
    let dt = TimeStep::new(0.01).unwrap();
    let solver = EulerMaruyama::new(params, dt);

    let mut rng_a = OxidizeRng::new(42);
    let mut rng_b = OxidizeRng::new(42);
    let mut rng_c = OxidizeRng::new(99);

    let path_a = solver.simulate(0.4, 32, &mut rng_a);
    let path_b = solver.simulate(0.4, 32, &mut rng_b);
    let path_c = solver.simulate(0.4, 32, &mut rng_c);

    assert_eq!(path_a, path_b);
    assert_ne!(path_a, path_c);

    let mut digest = TrajectoryDigest::new();
    digest.update_f64_slice(&path_a);
    assert_eq!(digest.finish_hex(), PINNED_OU_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_ctmc_trajectory_pinned_digest() {
    #[cfg(not(target_arch = "wasm32"))]
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let generator =
        DMatrix::from_row_slice(3, 3, &[-1.5, 1.0, 0.5, 0.6, -1.2, 0.6, 0.4, 0.8, -1.2]);
    let chain = ContinuousMarkovChain::new(generator).unwrap();

    let run = |seed: u64| -> (Vec<(f64, usize)>, String) {
        let mut rng = OxidizeRng::new(seed);
        let traj = chain.simulate_trajectory(0, 4.0, &mut rng).unwrap();
        let mut digest = TrajectoryDigest::new();
        for &(t, s) in &traj {
            digest.update_f64(t);
            digest.update_usize(s);
        }
        (traj, digest.finish_hex())
    };

    let (tr1, h1) = run(77);
    let (tr2, h2) = run(77);
    let (tr3, h3) = run(88);
    assert_eq!(tr1, tr2);
    assert_eq!(h1, h2);
    assert_ne!(tr1, tr3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_CTMC_DIGEST);
}
