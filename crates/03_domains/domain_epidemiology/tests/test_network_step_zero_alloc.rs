//! Zero-allocation and live topology builder tests for `NetworkEpidemicModel`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use domain_epidemiology::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use oxidize_core::rng::OxidizeRng;
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ALLOC_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_network_epidemic_builder_step_zero_alloc_and_bridge_r0() {
    let mut adj = vec![Vec::new(); 12];
    for i in 0..6 {
        for j in (i + 1)..6 {
            adj[i].push(j);
            adj[j].push(i);
            adj[i + 6].push(j + 6);
            adj[j + 6].push(i + 6);
        }
    }
    let positions: Vec<[f32; 2]> = (0..12).map(|i| [i as f32 * 10.0, 0.0]).collect();
    let mut states = vec![NodeState::Susceptible; 12];
    states[0] = NodeState::Infected;

    let unbridged = NetworkEpidemicModel::builder()
        .num_nodes(12)
        .parameters(0.35, 0.1)
        .states(states.clone())
        .positions(positions.clone())
        .adjacency(adj.clone())
        .build()
        .expect("valid custom network epidemic model");
    let r0_before = unbridged.effective_r0();

    for target in 6..12 {
        adj[0].push(target);
        adj[target].push(0);
    }
    let mut bridged = NetworkEpidemicModel::builder()
        .num_nodes(12)
        .parameters(0.35, 0.1)
        .states(states)
        .positions(positions)
        .adjacency(adj)
        .build()
        .expect("valid bridged network epidemic model");
    let r0_after = bridged.effective_r0();
    assert!(
        r0_after > r0_before,
        "Hub/bridge edges must raise heterogeneous effective_r0: before={r0_before}, after={r0_after}"
    );

    let mut rng = OxidizeRng::new(99);
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (total_active, _metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..50 {
            bridged.step_with_rng(&mut rng);
        }
        bridged.states().len()
    });
    assert_eq!(total_active, 12);
}
