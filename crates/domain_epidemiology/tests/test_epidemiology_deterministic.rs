//! Cross-platform deterministic tests for `domain_epidemiology` stochastic SIR and network models.

use domain_epidemiology::epidemiology::compartmental::SIRModel;
use domain_epidemiology::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use domain_epidemiology::epidemiology::stochastic::{GillespieSolver, probability_of_extinction};
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};

const PINNED_SIR_EXTINCTION_DIGEST: &str =
    "db51cb698668448aa767d3dfb9df710a800580cb3a921474e00a28bbf6889cde";
const PINNED_NETWORK_EPIDEMIC_DIGEST: &str =
    "e79f0fc080f5745dd68029124d0da61d31076e94156f019222503ba32b2e45bd";

#[test]
#[verified_engine::verified]
fn test_sir_gillespie_and_extinction_pinned_digest() {
    let model = SIRModel::new(120.0, 12.0, 1.8, 0.15).unwrap();
    let run = |seed: u64| -> (Vec<[f64; 4]>, String) {
        let mut solver = GillespieSolver::new(OxidizeRng::new(seed));
        let mut state = *model.state();
        let mut rows = Vec::with_capacity(30);
        let mut digest = TrajectoryDigest::new();
        for k in 1..=6 {
            digest.update_f64(probability_of_extinction(1.2 + (k as f64) * 0.35, k as f64));
        }
        for _ in 0..30 {
            let dt = solver.step(&model, &mut state).unwrap();
            let row = [dt, state.s, state.i, state.r];
            digest.update_f64_slice(&row);
            rows.push(row);
        }
        (rows, digest.finish_hex())
    };

    let (r1, h1) = run(42);
    let (r2, h2) = run(42);
    let (r3, h3) = run(43);
    assert_eq!(r1, r2);
    assert_eq!(h1, h2);
    assert_ne!(r1, r3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_SIR_EXTINCTION_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_network_epidemic_threaded_rng_pinned_digest() {
    let run = |seed: u64| -> (Vec<[usize; 3]>, String) {
        let mut rng = OxidizeRng::new(seed);
        let mut net = NetworkEpidemicModel::new_with_rng(28, 0.35, 0.12, &mut rng);
        let mut digest = TrajectoryDigest::new();
        for &[x, y] in net.positions() {
            digest.update_f32(x);
            digest.update_f32(y);
        }
        for &[u, v] in net.edges() {
            digest.update_usize(u);
            digest.update_usize(v);
        }
        let mut counts = Vec::with_capacity(16);
        for _ in 0..16 {
            net.step_with_rng(&mut rng);
            let s = net
                .states()
                .iter()
                .filter(|&&st| st == NodeState::Susceptible)
                .count();
            let i = net
                .states()
                .iter()
                .filter(|&&st| st == NodeState::Infected)
                .count();
            let r = net
                .states()
                .iter()
                .filter(|&&st| st == NodeState::Recovered)
                .count();
            digest.update_usize(s);
            digest.update_usize(i);
            digest.update_usize(r);
            counts.push([s, i, r]);
        }
        (counts, digest.finish_hex())
    };

    let (c1, h1) = run(101);
    let (c2, h2) = run(101);
    let (c3, h3) = run(202);
    assert_eq!(c1, c2);
    assert_eq!(h1, h2);
    assert_ne!(c1, c3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_NETWORK_EPIDEMIC_DIGEST);
}
