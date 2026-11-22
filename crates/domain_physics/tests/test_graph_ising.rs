#![allow(missing_docs)]
use approx::assert_relative_eq;
use domain_physics::physics::stat_mech::KB;
use domain_physics::physics::stat_mech::graph_ising::GraphSpinSystem;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn build_scale_free_star_web(num_nodes: usize) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for i in 1..num_nodes {
        // Connect every spoke to central hub 0 and secondary hub 1
        edges.push((0, i, 1.0));
        if i > 1 && i % 2 == 0 {
            edges.push((1, i, 0.75));
        }
        if i + 1 < num_nodes && i % 3 == 0 {
            edges.push((i, i + 1, -0.5));
        }
    }
    edges
}

#[test]
#[verified_engine::verified]
fn test_odd_cycle_triangle_geometric_frustration() {
    let edges = vec![(0, 1, -1.0), (1, 2, -1.0), (2, 0, -1.0)];
    let mut sys = GraphSpinSystem::new_with_spins(vec![1, 1, 1], &edges, Some(7));

    // All +1 on antiferromagnetic triangle frustrates all 3 bonds: H = -1 * (-3) = +3
    assert_eq!(sys.frustrated_edge_count(1.0), 3);
    assert_relative_eq!(sys.hamiltonian(1.0, 0.0), 3.0, epsilon = 1e-12);

    // Cool to ground state: odd cycle retains exactly 1 frustrated bond with H = -1
    sys.evolve(2_000, 0.15 / KB, 1.0, 0.0);
    assert_eq!(sys.frustrated_edge_count(1.0), 1);
    assert_relative_eq!(sys.hamiltonian(1.0, 0.0), -1.0, epsilon = 1e-12);
    assert_eq!(sys.magnetization().abs(), 1);
}

#[test]
#[verified_engine::verified]
fn test_ferromagnetic_hub_spoke_alignment() {
    let mut edges = Vec::new();
    for spoke in 1..16 {
        edges.push((0, spoke, 1.5));
        edges.push((spoke, (spoke % 15) + 1, 1.0));
    }
    let mut sys = GraphSpinSystem::new(16, &edges, Some(1234));
    sys.evolve(20_000, 0.35 / KB, 1.0, 0.05);

    assert!(sys.mean_magnetization().abs() > 0.85);
    assert_eq!(sys.frustrated_edge_count(1.0), 0);
}

#[test]
#[verified_engine::verified]
fn test_graph_spin_system_zero_allocations_odd_cycle_and_scale_free() {
    // 5-wheel odd-cycle frustrated graph
    let odd_wheel_edges = vec![
        (0, 1, -1.0),
        (0, 2, -1.0),
        (0, 3, -1.0),
        (0, 4, -1.0),
        (1, 2, -1.0),
        (2, 3, -1.0),
        (3, 4, -1.0),
        (4, 1, -1.0),
    ];
    let mut odd_sys = GraphSpinSystem::new(5, &odd_wheel_edges, Some(99));

    let sf_edges = build_scale_free_star_web(64);
    let mut sf_sys = GraphSpinSystem::new(64, &sf_edges, Some(2026));

    let temp = 1.2 / KB;
    let ((e_odd, m_sf, frust_odd), _metrics) = VerifiedEngine::run_verified(|| {
        odd_sys.evolve(500, temp, 1.0, 0.0);
        sf_sys.evolve(1_000, temp, 1.0, 0.1);
        (
            odd_sys.hamiltonian(1.0, 0.0),
            sf_sys.mean_magnetization(),
            odd_sys.frustrated_edge_count(1.0),
        )
    });

    assert!(e_odd.is_finite());
    assert!(m_sf.is_finite() && (-1.0..=1.0).contains(&m_sf));
    assert!(frust_odd >= 2);
}
