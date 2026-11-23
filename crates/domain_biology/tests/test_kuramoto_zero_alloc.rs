//! Zero-allocation and phase-transition tests for `KuramotoNetwork` and `HodgkinHuxleyNeuron`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use domain_biology::biology::neuroscience::{HodgkinHuxleyNeuron, KuramotoNetwork};
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ALLOC_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_kuramoto_and_hodgkin_huxley_steps_allocate_zero_heap_bytes() {
    let mut edges = Vec::new();
    for i in 0..16 {
        edges.push((i, (i + 1) % 16, 1.0));
        edges.push((i, (i + 3) % 16, 0.5));
    }
    let mut kuramoto = KuramotoNetwork::new(16, 4.5);
    kuramoto.reconfigure_topology(16, &edges, false, 0.4);

    let mut neurons: Vec<HodgkinHuxleyNeuron> =
        (0..8).map(|_| HodgkinHuxleyNeuron::new(-65.0)).collect();
    let mut scratch_inputs = [0.0_f64; 8];

    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (r, _metrics) = VerifiedEngine::run_verified(|| {
        for step in 0..64 {
            kuramoto.step(0.02);
            scratch_inputs.fill(0.0);
            if step < 10 {
                scratch_inputs[0] = 12.0;
            }
            for i in 0..neurons.len() {
                if neurons[i].v() > 0.0 {
                    scratch_inputs[(i + 1) % neurons.len()] += 15.0;
                }
            }
            for (i, neuron) in neurons.iter_mut().enumerate() {
                neuron.update(0.05, scratch_inputs[i]);
            }
        }
        kuramoto.order_parameter().0
    });

    assert!(r.is_finite() && (0.0..=1.0).contains(&r));
}

#[test]
fn test_kuramoto_synchronization_transition_and_bridge_edge_coupling() {
    let n = 10;
    let mut all_to_all = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            all_to_all.push((i, j, 1.0));
        }
    }

    let mut uncoupled = KuramotoNetwork::new(n, 0.0);
    uncoupled.reconfigure_topology(n, &all_to_all, false, 0.8);
    for _ in 0..400 {
        uncoupled.step(0.05);
    }
    let (r_uncoupled, _) = uncoupled.order_parameter();

    let mut locked = KuramotoNetwork::new(n, 6.0);
    locked.reconfigure_topology(n, &all_to_all, false, 0.8);
    for _ in 0..400 {
        locked.step(0.05);
    }
    let (r_locked, _) = locked.order_parameter();
    assert!(
        r_locked > 0.85 && r_locked > r_uncoupled + 0.35,
        "Strong coupling must phase-lock oscillators: r_locked={r_locked}, r_uncoupled={r_uncoupled}"
    );

    // Two cliques separated by counter-phase initialization synchronize once bridged.
    let mut disconnected = Vec::new();
    for i in 0..4 {
        for j in (i + 1)..4 {
            disconnected.push((i, j, 1.0));
            disconnected.push((i + 4, j + 4, 1.0));
        }
    }
    let mut split_net = KuramotoNetwork::new(8, 5.0);
    split_net.reconfigure_topology(8, &disconnected, false, 0.0);
    for i in 0..4 {
        split_net.set_phase(i, 0.2);
        split_net.set_phase(i + 4, 0.2 + std::f64::consts::PI);
    }
    for _ in 0..250 {
        split_net.step(0.05);
    }
    let (r_split, _) = split_net.order_parameter();
    assert!(
        r_split < 0.2,
        "Disconnected antiphase clusters must remain separated: r_split={r_split}"
    );

    let mut bridged = disconnected;
    bridged.push((0, 4, 3.0));
    bridged.push((1, 5, 3.0));
    split_net.set_phase(4, 0.2 + std::f64::consts::PI - 0.25);
    split_net.reconfigure_topology(8, &bridged, false, 0.0);
    for _ in 0..400 {
        split_net.step(0.05);
    }
    let (r_bridged, _) = split_net.order_parameter();
    assert!(
        r_bridged > 0.85,
        "Adding bridge edges must synchronize clusters: r_bridged={r_bridged}"
    );
}
