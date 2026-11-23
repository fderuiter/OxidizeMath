use super::*;

#[test]
#[verified_engine::verified]
fn test_kuramoto_step_rk4_zero_heap_allocations_under_verified_allocator() {
    let mut sys = KuramotoSystem::builder()
        .num_oscillators(16)
        .coupling_k(5.0)
        .frequency_spread(0.6)
        .build();

    // Everything inside #[verified_engine::verified] executes with VerifiedAllocator locked!
    for _ in 0..64 {
        sys.step_rk4(0.02);
        let (r, psi) = sys.order_parameter();
        assert!((0.0..=1.0).contains(&r));
        assert!(psi.is_finite());
    }
    assert!(sys.critical_coupling_estimate().is_finite());
}

#[test]
#[verified_engine::verified]
fn test_kuramoto_synchronizes_above_critical_coupling_and_drops_on_bridge_cut() {
    // Two 3-node triangles linked by a single bridge edge (2, 3).
    let connected_edges = vec![
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (3, 5, 1.0),
    ];
    let mut connected = KuramotoSystem::builder()
        .num_oscillators(6)
        .coupling_k(18.0)
        .frequency_spread(0.5)
        .edges(connected_edges)
        .build();
    for _ in 0..400 {
        connected.step_rk4(0.05);
    }
    let (r_connected, _) = connected.order_parameter();
    assert!(
        r_connected > 0.85,
        "Strong coupling across bridge should synchronize network, got R={r_connected}"
    );

    // Cut bridge edge (2, 3): clusters drift apart at opposing natural frequencies.
    let severed_edges = vec![
        (0, 1, 1.0),
        (1, 2, 1.0),
        (0, 2, 1.0),
        (3, 4, 1.0),
        (4, 5, 1.0),
        (3, 5, 1.0),
    ];
    connected.set_topology(6, &severed_edges);
    let mut min_r = 1.0_f64;
    for _ in 0..120 {
        connected.step_rk4(0.05);
        let (r, _) = connected.order_parameter();
        min_r = min_r.min(r);
    }
    assert!(
        min_r < 0.35,
        "Cutting bridge edge must desynchronize opposing clusters, got min_r={min_r}"
    );
}
