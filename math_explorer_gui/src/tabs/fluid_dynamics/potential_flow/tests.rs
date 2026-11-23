#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

fn run_frame(ctx: &egui::Context, tool: &mut PotentialFlowTool) {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(960.0, 640.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |c| tool.show(c));
}

#[test]
fn potential_flow_prebuilds_default_buffers_and_reuses_across_idle_frames() {
    let mut tool = PotentialFlowTool::default();
    assert_eq!(tool.grid_revision, 1);
    assert_eq!(tool.cached_revision, 1);
    assert!(!tool.origins.is_empty());
    assert_eq!(tool.origins.len(), tool.tips.len());
    assert!(tool.origins.capacity() >= (tool.grid_size + 1).pow(2));
    assert!(tool.tips.capacity() >= (tool.grid_size + 1).pow(2));

    {
        let borrowed_origins = PlotPoints::Borrowed(&tool.origins);
        let borrowed_tips = PlotPoints::Borrowed(&tool.tips);
        assert!(matches!(borrowed_origins, PlotPoints::Borrowed(_)));
        assert!(matches!(borrowed_tips, PlotPoints::Borrowed(_)));
        assert_eq!(borrowed_origins.points().len(), tool.origins.len());
    }

    let origins_ptr = tool.origins.as_ptr();
    let tips_ptr = tool.tips.as_ptr();
    let origins_cap = tool.origins.capacity();
    let ctx = egui::Context::default();
    run_frame(&ctx, &mut tool);
    run_frame(&ctx, &mut tool);

    assert_eq!(tool.grid_revision, 1);
    assert_eq!(tool.cached_revision, 1);
    assert_eq!(tool.origins.as_ptr(), origins_ptr);
    assert_eq!(tool.tips.as_ptr(), tips_ptr);
    assert_eq!(tool.origins.capacity(), origins_cap);
}

#[test]
fn potential_flow_clamps_high_grid_state_and_invalidates_on_load_add_and_clear() {
    let ctx = egui::Context::default();
    let mut tool = PotentialFlowTool::default();
    let high_grid = r#"{"elements":[{"element_type":"Uniform","strength":10.0,"x":0.0,"y":0.0,"angle":0.0}],"grid_size":1000,"view_range":10.0}"#;
    tool.try_load_state(high_grid)
        .expect("grid_size 1000 should load and clamp to interactive cap");
    assert_eq!(tool.grid_size, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
    assert!(tool.has_finite_parameters());
    assert_ne!(tool.cached_revision, tool.grid_revision);

    run_frame(&ctx, &mut tool);
    assert_eq!(tool.cached_revision, tool.grid_revision);
    assert_eq!(tool.origins.len(), (tool.grid_size + 1).pow(2));

    let rev_before_add = tool.grid_revision;
    tool.selected_element_type = ElementType::Vortex;
    tool.new_element_strength = 25.0;
    tool.add_element();
    assert!(tool.grid_revision > rev_before_add);
    assert_ne!(tool.cached_revision, tool.grid_revision);
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.cached_revision, tool.grid_revision);

    let cap_before_clear = tool.origins.capacity();
    tool.clear();
    assert!(tool.origins.is_empty() && tool.tips.is_empty());
    assert_eq!(tool.origins.capacity(), cap_before_clear);
    assert_ne!(tool.cached_revision, tool.grid_revision);
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.cached_revision, tool.grid_revision);
    assert!(tool.origins.is_empty() && tool.tips.is_empty());
}

#[test]
fn potential_flow_grid_size_and_view_range_preserve_or_grow_capacity_cleanly() {
    let ctx = egui::Context::default();
    let mut tool = PotentialFlowTool {
        elements: vec![SerializedFlowElement {
            element_type: ElementType::Uniform,
            strength: 10.0,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        }],
        ..Default::default()
    };
    tool.rebuild_field();
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.origins.len(), (20_usize + 1).pow(2));

    tool.grid_size = 40;
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.cached_revision, tool.grid_revision);
    assert_eq!(tool.origins.len(), (40_usize + 1).pow(2));
    assert!(tool.origins.capacity() >= (40_usize + 1).pow(2));

    let grown_origins_ptr = tool.origins.as_ptr();
    let grown_tips_ptr = tool.tips.as_ptr();
    let grown_cap = tool.origins.capacity();

    // Shrinking grid_size clears and repopulates without reallocating.
    tool.grid_size = 12;
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.origins.len(), (12_usize + 1).pow(2));
    assert_eq!(tool.origins.as_ptr(), grown_origins_ptr);
    assert_eq!(tool.tips.as_ptr(), grown_tips_ptr);
    assert_eq!(tool.origins.capacity(), grown_cap);

    // Updating view_range invalidates and updates coordinates in-place.
    let prev_rev = tool.cached_revision;
    tool.view_range = 25.0;
    run_frame(&ctx, &mut tool);
    assert!(tool.cached_revision > prev_rev);
    assert!((tool.origins[0].x - (-25.0)).abs() < 1e-9);
    assert_eq!(tool.origins.as_ptr(), grown_origins_ptr);
}

#[test]
fn potential_flow_skips_source_singularity_and_rebuilds_in_place_without_allocating() {
    let mut tool = PotentialFlowTool {
        elements: vec![SerializedFlowElement {
            element_type: ElementType::Source,
            strength: 15.0,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        }],
        grid_size: 10,
        view_range: 5.0,
        ..Default::default()
    };
    tool.rebuild_field();
    tool.rebuild_arrows();

    // (0.0, 0.0) lands on the central grid node where Source::velocity returns zero.
    assert_eq!(tool.origins.len(), (10_usize + 1).pow(2) - 1);
    assert!(
        tool.origins
            .iter()
            .chain(tool.tips.iter())
            .all(|p| p.x.is_finite() && p.y.is_finite())
    );

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            tool.reconcile_arrows();
            tool.grid_size = 8;
            tool.reconcile_arrows();
            assert_eq!(tool.origins.len(), (8_usize + 1).pow(2) - 1);
        });
    }
}
