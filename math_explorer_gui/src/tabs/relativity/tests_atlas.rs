#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Desktop unit tests for 2D lensing atlas zero-allocation convergence, click-to-teleport,
//! URL round-trip persistence, and guided story / playable challenge registrations.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

use eframe::egui;

use super::tests::test_raw_input;
use super::*;
use crate::app::MathExplorerApp;
use crate::async_sim::SimulationRunner;
use crate::async_sim::atlas::{AtlasRenderRequest, AtlasRunner, AtlasTeleportEvent};
use crate::framework::InteractiveTool;
use crate::framework::state_guard::MAX_SNAPSHOT_BYTES;
use crate::starter_gallery::{get_goal_challenges_for_domain, get_stories_for_domain};
use crate::tabs::ExplorerTab;
use crate::url_state::{URL_JSON_BOUNDS, encode_share_fragment, parse_share_location};

#[test]
fn relativity_atlas_converges_zero_alloc_under_verified_allocator() {
    let mut tool = GeodesicPlaygroundTool::default();
    tool.import_preset_json(presets::PRESET_PHOTON_RING_LENSING)
        .unwrap();
    let latest_gen = Arc::new(AtomicU64::new(1));
    let shared = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 32,
        height: 32,
        high_contrast: false,
        cartographer: Arc::new(tool.build_cartographer(("impact_or_momentum", "spin_chi"))),
    }));
    let mut runner = AtlasRunner::new(shared, latest_gen);
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..64 {
                runner.step();
                if runner.is_complete() {
                    break;
                }
            }
        });
    }
    assert!(runner.is_complete());
}

#[test]
fn relativity_atlas_teleports_and_roundtrips_url_state() {
    let mut tool = GeodesicPlaygroundTool::default();
    tool.import_preset_json(presets::PRESET_PHOTON_RING_LENSING)
        .unwrap();
    tool.atlas.set_visible(true);
    tool.state.atlas_visible = true;
    tool.atlas
        .set_axis_keys("impact_or_momentum", "spin_chi")
        .unwrap();
    tool.state.atlas_axes = ["impact_or_momentum".into(), "spin_chi".into()];
    let ctx = egui::Context::default();
    for _ in 0..90 {
        let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
        });
        if tool.atlas.is_converged() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(tool.atlas.current_stride(), 1);

    let b_ring = playground::kerr_prograde_critical_b(2.0, 0.8) + 0.04;
    tool.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "impact_or_momentum",
        x_val: b_ring,
        y_key: "spin_chi",
        y_val: 0.8,
        regime_label: "Photon-Ring Winding / Rosette",
    });
    assert_eq!(tool.state.spacetime, SpacetimePreset::KerrEquatorial);
    assert!((tool.state.impact_or_momentum - b_ring).abs() < 1e-9);

    tool.sandbox
        .lapse_bar
        .set_expression("1 - Rs / r + q^2 / r^2");
    tool.sync_sandbox_to_state();
    let url_json = tool.save_url_state().unwrap();
    URL_JSON_BOUNDS
        .validate_json(&url_json)
        .expect("within URL_JSON_BOUNDS");
    let frag = encode_share_fragment(
        "relativity",
        Some("GeodesicPlaygroundTool"),
        Some(&url_json),
    )
    .unwrap();
    let parsed = parse_share_location(&frag, "").unwrap().unwrap();
    let mut restored = GeodesicPlaygroundTool::default();
    restored
        .try_load_state(parsed.state_json.as_deref().unwrap())
        .unwrap();
    assert_eq!(restored.state, tool.state);

    let mut tab = RelativityTab::default();
    assert!(tab.select_tool_by_name("GeodesicPlaygroundTool"));
    let fw = tab.framework_mut().unwrap();
    fw.cache_valid_snapshot();
    assert!(fw.last_valid_snapshot.as_ref().unwrap().memory_size() <= MAX_SNAPSHOT_BYTES);
}

fn find_atlas_click_pos(out: &egui::FullOutput) -> egui::Pos2 {
    let b = out
        .platform_output
        .accesskit_update
        .as_ref()
        .and_then(|u| {
            u.nodes.iter().find_map(|(_, n)| {
                (n.role() == egui::accesskit::Role::Canvas
                    && n.label() == Some("2D Parameter Regime Atlas"))
                .then(|| n.bounds())
                .flatten()
            })
        })
        .expect("2D Parameter Regime Atlas canvas bounds");
    egui::pos2(
        (b.x0 + 0.686 * (b.x1 - b.x0)) as f32,
        ((b.y0 + b.y1) * 0.5) as f32,
    )
}

#[test]
fn relativity_atlas_canvas_click_updates_state_and_syncs_url_fragment() {
    use crate::url_state::{URL_SYNC_DEBOUNCE_MS, UrlSyncController};
    use eframe::App as _;

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut frame = eframe::Frame::_new_kittest();
    let mut app = MathExplorerApp::default();
    app.apply_share_location_strings("#tab=relativity&tool=GeodesicPlaygroundTool", "", None)
        .unwrap();
    {
        let fw = app.tabs[app.selected_tab].framework_mut().unwrap();
        let tool = fw.active_tool.as_mut().unwrap().as_any_mut().unwrap();
        let tool = tool.downcast_mut::<GeodesicPlaygroundTool>().unwrap();
        tool.import_preset_json(presets::PRESET_PHOTON_RING_LENSING)
            .unwrap();
        tool.atlas.set_visible(true);
        tool.state.atlas_visible = true;
    }
    let out0 = ctx.run(test_raw_input(Vec::new()), |c| app.update(c, &mut frame));
    let pos = find_atlas_click_pos(&out0);
    let mut sync = UrlSyncController::default();
    let base_sig = app.current_url_signature();
    sync.mark_synced(base_sig, Some(app.encode_current_share_fragment().unwrap()));
    let evs = vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(test_raw_input(evs), |c| app.update(c, &mut frame));
    let next_sig = app.current_url_signature();
    assert_ne!(next_sig, base_sig, "clicking atlas must bump URL signature");
    let _ = sync.poll_at_millis(next_sig, false, 100, URL_SYNC_DEBOUNCE_MS, || {
        app.encode_current_share_fragment()
    });
    let _ = sync.poll_at_millis(
        next_sig,
        false,
        100 + URL_SYNC_DEBOUNCE_MS + 20,
        URL_SYNC_DEBOUNCE_MS,
        || app.encode_current_share_fragment(),
    );
    assert!(
        sync.last_synced_fragment()
            .unwrap()
            .starts_with("#tab=relativity&tool=GeodesicPlaygroundTool&state=")
    );
}

#[test]
fn relativity_stories_register_and_step_cleanly() {
    let stories = get_stories_for_domain("relativity");
    assert!(
        !stories.is_empty(),
        "expected >= 1 relativity PresetStoryMetadata"
    );

    let mut tab = RelativityTab::default();
    let fw = tab.framework_mut().unwrap();
    fw.launch_story(stories[0], 0).unwrap();
    for step_idx in 1..stories[0].steps.len() {
        fw.step_story(step_idx).unwrap();
        let tool = fw
            .active_tool
            .as_ref()
            .unwrap()
            .as_any()
            .unwrap()
            .downcast_ref::<GeodesicPlaygroundTool>()
            .unwrap();
        assert!(!tool.samples.is_empty());
    }
}

#[test]
fn relativity_goal_challenges_register_and_solve_cleanly() {
    let challenges = get_goal_challenges_for_domain("relativity");
    assert!(
        challenges.len() >= 2,
        "expected >= 2 relativity GoalChallengeMetadata"
    );

    let mut tab = RelativityTab::default();
    let fw = tab.framework_mut().unwrap();

    fw.launch_challenge_by_id("relativity_photon_ring_bifurcation", None)
        .unwrap();
    assert_eq!(fw.evaluate_challenge_predicates(None), Some(false));
    {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<GeodesicPlaygroundTool>()
            .unwrap();
        tool.state.impact_or_momentum = 5.22;
        tool.clamp_parameters();
    }
    assert_eq!(fw.evaluate_challenge_predicates(None), Some(true));

    fw.launch_challenge_by_id("relativity_kerr_ergosphere_skim", None)
        .unwrap();
    assert_eq!(fw.evaluate_challenge_predicates(None), Some(false));
    {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<GeodesicPlaygroundTool>()
            .unwrap();
        tool.apply_atlas_teleport(AtlasTeleportEvent {
            x_key: "impact_or_momentum",
            x_val: 2.96,
            y_key: "spin_chi",
            y_val: 0.9,
            regime_label: "Photon-Ring Winding / Rosette",
        });
    }
    assert_eq!(fw.evaluate_challenge_predicates(None), Some(true));
}
