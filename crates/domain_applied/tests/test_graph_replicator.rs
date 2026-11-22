#![allow(missing_docs)]
use approx::assert_relative_eq;
use domain_applied::applied::game_theory::evolutionary::{
    GraphReplicatorSystem, ReplicatorDynamics,
};
use nalgebra::{DMatrix, DVector};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn build_hub_cooperator_network() -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    // Cooperator core (0 is hub connected to cooperators 1..5 and defector leaves 6..9)
    let mut edges = vec![
        (0, 1, 1.5),
        (0, 2, 1.5),
        (0, 3, 1.5),
        (0, 4, 1.5),
        (0, 5, 1.5),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (5, 1, 1.0),
    ];
    for leaf in 6..10 {
        edges.push((0, leaf, 1.0));
    }

    let mut states = Vec::with_capacity(20);
    for node in 0..10 {
        if node <= 5 {
            states.extend_from_slice(&[0.95, 0.05]);
        } else {
            states.extend_from_slice(&[0.05, 0.95]);
        }
    }
    (edges, states)
}

#[test]
#[verified_engine::verified]
fn test_spatial_cooperators_survive_while_panmictic_defects() {
    // Prisoner's Dilemma: R=3.0, S=0.0, T=4.2, P=1.0
    let payoff = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 4.2, 1.0]);
    let (edges, states) = build_hub_cooperator_network();

    let mut spatial = GraphReplicatorSystem::new(10, &payoff, &edges, &states).unwrap();
    let mut init_mean = [0.0; 2];
    spatial.mean_frequencies(&mut init_mean);

    // Well-mixed replicator from the exact same initial mean collapses cooperation toward 0
    let panmictic = ReplicatorDynamics::new(payoff.clone()).unwrap();
    let traj = panmictic.simulate(DVector::from_vec(init_mean.to_vec()), 15.0, 0.05);
    let final_panmictic_c = traj.last().unwrap().1[0];
    assert!(final_panmictic_c < 0.05);

    // Spatial hub-cluster reciprocity increases cooperation above initial mean
    spatial.evolve(300, 0.05);
    let mut final_spatial = [0.0; 2];
    spatial.mean_frequencies(&mut final_spatial);
    assert!(final_spatial[0] > init_mean[0]);
    assert!(final_spatial[0] > 0.8);
    assert_relative_eq!(final_spatial[0] + final_spatial[1], 1.0, epsilon = 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_negative_weights_quench_spatial_hub_cooperation() {
    let payoff = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 4.2, 1.0]);
    let (mut edges, states) = build_hub_cooperator_network();
    for edge in &mut edges {
        edge.2 = -edge.2.abs();
    }

    let mut quenched = GraphReplicatorSystem::new(10, &payoff, &edges, &states).unwrap();
    let mut init_mean = [0.0; 2];
    quenched.mean_frequencies(&mut init_mean);

    quenched.evolve(300, 0.05);
    let mut after_mean = [0.0; 2];
    quenched.mean_frequencies(&mut after_mean);
    assert!(after_mean[0] < init_mean[0] + 0.05);
    assert_relative_eq!(after_mean[0] + after_mean[1], 1.0, epsilon = 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_graph_replicator_zero_allocations_odd_cycle_and_scale_free() {
    let payoff = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 4.0, 1.0]);

    // Odd-cycle triangle
    let triangle_edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, -0.5)];
    let mut tri_sys =
        GraphReplicatorSystem::from_population_mean(3, &payoff, &triangle_edges, &[0.5, 0.5])
            .unwrap();

    // Scale-free hub-and-spoke web
    let mut sf_edges = Vec::new();
    for i in 1..48 {
        sf_edges.push((0, i, 1.2));
        if i > 1 && i % 2 == 0 {
            sf_edges.push((1, i, 0.8));
        }
        if i + 1 < 48 && i % 3 == 0 {
            sf_edges.push((i, i + 1, 0.5));
        }
    }
    let mut sf_sys =
        GraphReplicatorSystem::from_population_mean(48, &payoff, &sf_edges, &[0.55, 0.45]).unwrap();

    let mut mean_tri = [0.0_f64; 2];
    let mut mean_sf = [0.0_f64; 2];

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        tri_sys.evolve(100, 0.05);
        sf_sys.evolve(100, 0.05);
        tri_sys.mean_frequencies(&mut mean_tri);
        sf_sys.mean_frequencies(&mut mean_sf);
    });

    assert_relative_eq!(mean_tri[0] + mean_tri[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(mean_sf[0] + mean_sf[1], 1.0, epsilon = 1e-12);
}
