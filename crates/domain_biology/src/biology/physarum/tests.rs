use super::*;
use oxidize_core::rng::OxidizeRng;

#[test]
#[verified_engine::verified]
fn test_physarum_bounds_and_bit_identical_rng_determinism() {
    let clamped = PhysarumEngine::new(
        16,
        999,
        10,
        PhysarumParams {
            sensor_angle: f64::NAN,
            turn_angle: 99.0,
            sensor_distance: -4.0,
            step_size: f64::INFINITY,
            deposition: 0.0,
            diffusion_rate: 2.0,
            decay_rate: -1.0,
            prune_rate: 9.0,
            beacon_strength: -3.0,
        },
        OxidizeRng::new(11),
    );
    assert_eq!(clamped.width(), MIN_PHYSARUM_GRID);
    assert_eq!(clamped.height(), MAX_PHYSARUM_GRID);
    assert_eq!(clamped.agent_count(), MIN_PHYSARUM_AGENTS);
    assert!(clamped.params().is_within_bounds());

    let mut eng_a = PhysarumEngine::from_preset(
        96,
        96,
        2048,
        PhysarumPresetKind::TokyoRail,
        OxidizeRng::new(2026),
    );
    let mut eng_b = PhysarumEngine::from_preset(
        96,
        96,
        2048,
        PhysarumPresetKind::TokyoRail,
        OxidizeRng::new(2026),
    );
    let mut eng_c = PhysarumEngine::from_preset(
        96,
        96,
        2048,
        PhysarumPresetKind::TokyoRail,
        OxidizeRng::new(2027),
    );

    for _ in 0..60 {
        eng_a.step();
        eng_b.step();
        eng_c.step();
    }

    assert_eq!(eng_a.trail(), eng_b.trail());
    assert_eq!(eng_a.agent_x(), eng_b.agent_x());
    assert_eq!(eng_a.agent_y(), eng_b.agent_y());
    assert_eq!(eng_a.agent_heading(), eng_b.agent_heading());
    assert_ne!(eng_a.trail(), eng_c.trail());
}

#[test]
#[verified_engine::verified]
fn test_asymmetric_maze_routes_over_90_pct_trail_to_short_corridor_within_450_steps() {
    let mut engine = PhysarumEngine::from_preset(
        128,
        128,
        4096,
        PhysarumPresetKind::AsymmetricMaze,
        OxidizeRng::new(42),
    );
    let corridors = asymmetric_maze_corridors(engine.width(), engine.height());

    for _ in 0..420 {
        engine.step();
    }

    let [sx0, sy0, sx1, sy1] = corridors.short_branch;
    let [lx0, ly0, lx1, ly1] = corridors.long_branch;
    let short_mass = engine.trail_mass_in_rect(sx0, sy0, sx1, sy1);
    let long_mass = engine.trail_mass_in_rect(lx0, ly0, lx1, ly1);
    let total_corridor_mass = short_mass + long_mass;
    assert!(total_corridor_mass > 10.0);
    let short_share = short_mass / total_corridor_mass;
    assert!(
        short_share >= 0.90,
        "Expected >= 90% trail mass in short corridor within 450 steps, got {:.4} (short={:.2}, long={:.2})",
        short_share,
        short_mass,
        long_mass
    );

    let bench = extract_physarum_benchmark(&engine);
    assert_eq!(bench.connected_pair_fraction, 1.0);
    assert!(bench.mean_route_stretch >= 1.0 && bench.mean_route_stretch < 1.25);
}

#[test]
#[verified_engine::verified]
fn test_three_terminal_steiner_forms_y_junction_with_over_12_pct_length_reduction() {
    let mut engine = PhysarumEngine::from_preset(
        128,
        128,
        6144,
        PhysarumPresetKind::SteinerTriangle,
        OxidizeRng::new(42),
    );

    for _ in 0..340 {
        engine.step();
    }

    let bench = extract_physarum_benchmark(&engine);
    assert_eq!(
        bench.connected_pair_fraction, 1.0,
        "All 3 Steiner terminals must remain connected"
    );
    assert!(
        !bench.steiner_junctions.is_empty(),
        "Expected at least one emergent central Steiner Y-junction"
    );
    assert!(
        bench.length_reduction_pct >= 12.0,
        "Expected >= 12.0% wire length reduction below static 2-edge MST, got {:.2}% (wire={:.2}, mst={:.2}, junctions={:?})",
        bench.length_reduction_pct,
        bench.wire_cost,
        bench.mst_cost,
        bench.steiner_junctions
    );
}

#[test]
#[verified_engine::verified]
fn test_tokyo_rail_fault_tolerance_and_severed_artery_healing() {
    let mut engine = PhysarumEngine::from_preset(
        128,
        128,
        6144,
        PhysarumPresetKind::TokyoRail,
        OxidizeRng::new(77),
    );
    for _ in 0..240 {
        engine.step();
    }
    let before = extract_physarum_benchmark(&engine);
    assert!(
        before.connected_pair_fraction >= 0.90,
        "Expected >= 0.90 connected_pair_fraction in TokyoRail, got {:.3} (edges={})",
        before.connected_pair_fraction,
        before.edges.len()
    );
    assert!(
        before.pairwise_fault_survival > 0.50,
        "Tokyo-rail mesh should survive single-edge faults: pairwise={:.3}",
        before.pairwise_fault_survival
    );

    // Sever a major central artery between Shinjuku and Tokyo Central and verify self-healing
    engine.sever_trail_disc(60, 63, 5.0);
    for _ in 0..140 {
        engine.step();
    }
    let healed = extract_physarum_benchmark(&engine);
    assert!(
        healed.connected_pair_fraction >= 0.90,
        "Colony should reroute/heal severed artery, connected_pair_fraction={:.3}",
        healed.connected_pair_fraction
    );

    // Verify terminal toggle and obstacle brush bounds
    let initial_terms = engine.terminal_count();
    assert!(engine.toggle_terminal_at(24.0, 24.0));
    assert_eq!(engine.terminal_count(), initial_terms + 1);
    engine.paint_obstacle_disc(24, 24, 4.0, true);
    assert_eq!(
        engine.obstacles()
            [math_commons::math_kernel::types::flatten_2d_index(24, 24, engine.width())],
        0,
        "Obstacles must remain cleared directly under active nutrient beacons"
    );
}
