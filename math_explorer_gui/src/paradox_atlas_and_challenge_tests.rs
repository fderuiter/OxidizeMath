//! End-to-end acceptance tests for Parrondo `ToolMetadata` disambiguation, legacy alias
//! routing, zero-allocation 2D paradox cartographers, `CalloutManager` teleportation,
//! challenge-mode spoiler gating, and `math_commons::ast` (`CompiledExpr`) stories/challenges.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::app::MathExplorerApp;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::command_palette::CommandPalette;
use crate::framework::{InteractiveTool, SimulationFramework, ToolMetadata};
use crate::starter_gallery::{GoalChallengeMetadata, PresetStoryMetadata};
use crate::tabs::clinical_trials::randomization::RandomizationTool;
use crate::tabs::clinical_trials::simpson_atlas::{simpson_theory_parameters, SimpsonCartographer};
use crate::tabs::clinical_trials::simpson_paradox::SimpsonParadoxLab;
use crate::tabs::financial_math::parrondo::atlas::ParrondoKellyCartographer;
use crate::tabs::financial_math::parrondo::ParrondoParadoxTool as KellyParrondoTool;
use crate::tabs::paradox_lab::braess::{BraessNetworkParadoxTool, BraessState};
use crate::tabs::paradox_lab::braess_atlas::BraessCartographer;
use crate::tabs::paradox_lab::parrondo::{ParrondoJumpRatchetTool, ParrondoVariant};
use crate::tabs::paradox_lab::parrondo_atlas::ParrondoJumpCartographer;
use eframe::egui;
use math_explorer::pure_math::statistics::markov::parrondo::ParrondoSchedule;
use scientific_metadata::citation_registry::CitationRegistry;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashSet;

fn contrast_on_black(c: egui::Color32) -> f64 {
    let lin = |u: u8| {
        let s = f64::from(u) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let lum = 0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b());
    (lum + 0.05) / 0.05
}

fn assert_cartographer_zero_alloc_and_contrast<C: RegimeCartographer>(cart: &C) {
    let axes = cart.axes();
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for sy in 0..32 {
            for sx in 0..32 {
                let (p1, p2) = axes.pixel_to_coords(sx, sy, 32, 32);
                let cell = cart.evaluate_cell(p1, p2);
                assert!(cell.primary.is_finite() && cell.secondary.is_finite());
                let hc = cart.color_for_cell(cell, true);
                assert!(contrast_on_black(hc) >= 3.0);
            }
        }
    });
}

#[test]
fn test_parrondo_metadata_command_palette_and_citation_disambiguation() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut ids = HashSet::new();
    let mut names = HashSet::new();
    for meta in inventory::iter::<ToolMetadata> {
        assert!(
            ids.insert(meta.id),
            "Duplicate ToolMetadata::id '{}'",
            meta.id
        );
        assert!(
            names.insert(meta.display_name),
            "Duplicate ToolMetadata::display_name '{}'",
            meta.display_name
        );
    }
    assert!(ids.contains("ParrondoParadoxTool") && ids.contains("ParrondoJumpRatchetTool"));
    assert!(
        names.contains("Parrondo's Kelly Ratchet") && names.contains("Parrondo's Jump Ratchet")
    );

    let mut fw_fin = SimulationFramework::new("financial_math");
    assert!(fw_fin.select_tool_by_name("ParrondoParadoxTool"));
    let mut fw_px = SimulationFramework::new("paradox_lab");
    assert!(fw_px.select_tool_by_name("ParrondoJumpRatchetTool"));

    let kelly_cite = CitationRegistry::get("Parrondo's Kelly Ratchet").expect("Kelly citation");
    let jump_cite = CitationRegistry::get("Parrondo's Jump Ratchet").expect("Jump citation");
    assert_ne!(
        kelly_cite, jump_cite,
        "Parrondo citations must not overwrite each other"
    );
    assert!(kelly_cite.contains("Kelly") && jump_cite.contains("Harmer & Abbott"));

    let app = MathExplorerApp::default();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut palette = CommandPalette::default();
    palette.open();
    palette.query = "parrondo".to_string();
    let out = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = palette.show(ctx, &app.tabs);
    });
    let labels: Vec<String> = out
        .platform_output
        .accesskit_update
        .unwrap()
        .nodes
        .iter()
        .filter_map(|(_, n)| n.label().map(str::to_string))
        .collect();
    assert!(labels
        .iter()
        .any(|l| l.contains("Parrondo's Kelly Ratchet")));
    assert!(labels.iter().any(|l| l.contains("Parrondo's Jump Ratchet")));
}

#[test]
fn test_legacy_parrondo_url_alias_routing() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let fin_preset = r#"{"metadata":{"schema_version":"1.0","title":"Legacy Kelly","domain":"financial_math","timestamp":"2026-11-23T00:00:00Z"},"config":{"p_a_base":0.49,"p_b0_base":0.11,"p_b1_base":0.74,"epsilon":0.006,"modulus":3,"schedule":"AABB","num_walkers":128,"num_steps":120,"seed":7,"view_mode":"SplitOverview"}}"#;
    let px_preset = r#"{"metadata":{"schema_version":"1.0","title":"Legacy Jump","domain":"paradox_lab","timestamp":"2026-11-23T00:00:00Z"},"config":{"bias_epsilon":0.007,"mix_gamma":0.42,"jump_switch_rate":1.8,"variant":"HistoryDependent","focus_schedule":"RandomMix","steps":90,"walkers":128,"seed":9}}"#;
    let fin_hash = format!(
        "#tab=financial_math&tool=ParrondoParadoxTool&state={}",
        crate::url_state::encode_base64url(fin_preset.as_bytes())
    );
    let px_hash = format!(
        "#tab=paradox_lab&tool=ParrondoParadoxTool&state={}",
        crate::url_state::encode_base64url(px_preset.as_bytes())
    );

    let mut app = MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings(&px_hash, "", None)
        .unwrap());
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "paradox_lab");
    assert_eq!(
        app.tabs[app.selected_tab].active_tool_id(),
        Some("ParrondoJumpRatchetTool")
    );
    let px_restored = app.tabs[app.selected_tab]
        .export_url_preset_json()
        .unwrap()
        .unwrap();
    assert!(px_restored.contains("HistoryDependent") && px_restored.contains("0.42"));

    assert!(app
        .apply_share_location_strings(&fin_hash, "", None)
        .unwrap());
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "financial_math");
    assert_eq!(
        app.tabs[app.selected_tab].active_tool_id(),
        Some("ParrondoParadoxTool")
    );
    let fin_restored = app.tabs[app.selected_tab]
        .export_url_preset_json()
        .unwrap()
        .unwrap();
    assert!(fin_restored.contains("AABB") && fin_restored.contains("0.74"));
}

#[test]
fn test_legacy_parrondo_session_alias_routing() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let legacy_px_session = r#"{"selected_tool_index":1,"selected_tool_name":"Parrondo's Paradox","show_theory_portal":false,"tool_states":{"Parrondo's Paradox":{"bias_epsilon":0.009,"mix_gamma":0.33,"jump_switch_rate":1.5,"variant":"CapitalDependent","focus_schedule":"PeriodicABB","steps":80,"walkers":64,"seed":11}}}"#;
    let mut px_fw = SimulationFramework::new("paradox_lab");
    px_fw.load_state(legacy_px_session);
    assert_eq!(
        px_fw.active_tool.as_ref().map(|t| t.name()),
        Some("Parrondo's Jump Ratchet")
    );
    assert!(px_fw
        .active_tool
        .as_ref()
        .and_then(|t| t.save_state())
        .unwrap()
        .contains("0.33"));

    let legacy_fin_session = r#"{"selected_tool_index":0,"selected_tool_name":"Parrondo's Paradox","show_theory_portal":false,"tool_states":{"Parrondo's Paradox":{"p_a_base":0.49,"p_b0_base":0.11,"p_b1_base":0.74,"epsilon":0.006,"modulus":3,"schedule":"AABB","num_walkers":128,"num_steps":120,"seed":7,"view_mode":"SplitOverview"}}}"#;
    let mut fin_fw = SimulationFramework::new("financial_math");
    fin_fw.load_state(legacy_fin_session);
    assert_eq!(
        fin_fw.active_tool.as_ref().map(|t| t.name()),
        Some("Parrondo's Kelly Ratchet")
    );
    assert!(fin_fw
        .active_tool
        .as_ref()
        .and_then(|t| t.save_state())
        .unwrap()
        .contains("AABB"));
}

#[test]
fn test_all_four_paradox_cartographers_zero_alloc_and_high_contrast() {
    let kelly = KellyParrondoTool::default();
    let kelly_theory = kelly.theory_parameters();
    let sched = ParrondoSchedule::parse("ABB").unwrap();
    for pair in [("p_a_base", "epsilon"), ("p_b0_base", "p_b1_base")] {
        let cart = ParrondoKellyCartographer::new(
            &kelly_theory,
            pair,
            3,
            &sched,
            [0.50, 0.10, 0.75, 0.005],
        );
        assert_cartographer_zero_alloc_and_contrast(&cart);
    }

    let jump = ParrondoJumpRatchetTool::default();
    let jump_theory = jump.theory_parameters();
    for pair in [
        ("bias_epsilon", "mix_gamma"),
        ("bias_epsilon", "jump_switch_rate"),
    ] {
        let cart = ParrondoJumpCartographer::new(
            &jump_theory,
            pair,
            ParrondoVariant::CapitalDependent,
            0.005,
            0.50,
            1.2,
        );
        assert_cartographer_zero_alloc_and_contrast(&cart);
    }

    let braess = BraessNetworkParadoxTool::default();
    let braess_theory = braess.theory_parameters();
    for pair in [("demand", "shortcut_cost"), ("demand", "shortcut_toll")] {
        let cart = BraessCartographer::new(&braess_theory, pair, BraessState::default());
        assert_cartographer_zero_alloc_and_contrast(&cart);
    }

    let simpson_theory = simpson_theory_parameters();
    let simpson_cart = SimpsonCartographer::new(
        &simpson_theory,
        ("mild_treatment_share", "severe_treatment_share"),
        400,
    );
    assert_cartographer_zero_alloc_and_contrast(&simpson_cart);
}

#[test]
fn test_paradox_atlas_teleport_bounds_and_callout_sync() {
    let mut kelly = KellyParrondoTool::default();
    kelly.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "p_b0_base",
        x_val: 0.03,
        y_key: "p_b1_base",
        y_val: 0.95,
        regime_label: "Paradox Wedge",
    });
    kelly.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "p_a_base",
        x_val: 0.99,
        y_key: "epsilon",
        y_val: 0.045,
        regime_label: "Boundary Clamp",
    });
    let kelly_json = kelly.save_state().unwrap();
    assert!(kelly.has_finite_parameters() && kelly_json.contains("\"p_a_base\":0.65"));
    assert!(kelly
        .callouts
        .callouts()
        .iter()
        .any(|c| c.tag.as_deref() == Some("atlas_teleport")));

    let mut jump = ParrondoJumpRatchetTool::default();
    jump.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "bias_epsilon",
        x_val: 0.004,
        y_key: "jump_switch_rate",
        y_val: 9.0,
        regime_label: "Dual Ratchet",
    });
    assert!(
        jump.has_finite_parameters()
            && jump
                .observables()
                .iter()
                .any(|&(k, v)| k == "drift_abb" && v > 0.0)
    );

    let mut braess = BraessNetworkParadoxTool::default();
    braess.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "demand",
        x_val: 0.8,
        y_key: "shortcut_toll",
        y_val: 0.5,
        regime_label: "Pigou Social Optimum",
    });
    assert!((braess.metrics().price_of_anarchy - 1.0).abs() < 1e-4);

    let mut simpson = SimpsonParadoxLab::default();
    simpson.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "mild_treatment_share",
        x_val: 0.50,
        y_key: "severe_treatment_share",
        y_val: 0.50,
        regime_label: "Balanced Protective",
    });
    assert!(!simpson.is_simpson_reversal);
    assert!(simpson
        .callout_manager
        .callouts()
        .iter()
        .any(|c| c.tag.as_deref() == Some("atlas_teleport")));
}

fn find_goal_challenge(id: &str) -> &'static GoalChallengeMetadata {
    inventory::iter::<GoalChallengeMetadata>
        .into_iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("Missing GoalChallengeMetadata '{id}'"))
}

fn collect_button_labels(fw: &mut SimulationFramework, domain: &'static str) -> Vec<String> {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            fw.show_inside(ui, domain);
        });
    });
    let out = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = &mut frame;
            fw.show_inside(ui, domain);
        });
    });
    out.platform_output
        .accesskit_update
        .unwrap()
        .nodes
        .iter()
        .filter(|(_, n)| n.role() == egui::accesskit::Role::Button)
        .filter_map(|(_, n)| n.label().map(str::to_string))
        .collect()
}

#[test]
fn test_paradox_stories_and_ast_challenges_unsolved_to_solved() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for story_id in [
        "parrondo_kelly_ratchet_tour",
        "parrondo_jump_ratchet_tour",
        "braess_wheatstone_paradox_tour",
        "simpson_confounding_reversal_tour",
    ] {
        let story = inventory::iter::<PresetStoryMetadata>
            .into_iter()
            .find(|s| s.id == story_id)
            .unwrap_or_else(|| panic!("Missing story '{story_id}'"));
        let mut fw = SimulationFramework::new(story.domain);
        for idx in 0..story.steps.len() {
            fw.launch_story(story, idx).unwrap();
            assert!(!fw.active_tool.as_ref().unwrap().observables().is_empty());
        }
    }

    for (cid, solved_state) in [
        (
            "parrondo_kelly_wedge_hunt",
            r#"{"p_a_base":0.5,"p_b0_base":0.1,"p_b1_base":0.75,"epsilon":0.005,"modulus":3,"schedule":"ABB","num_walkers":512,"num_steps":240,"seed":42,"view_mode":"SplitOverview"}"#,
        ),
        (
            "parrondo_jump_ratchet_challenge",
            r#"{"bias_epsilon":0.005,"mix_gamma":0.5,"jump_switch_rate":1.2,"variant":"CapitalDependent","focus_schedule":"PeriodicABB","steps":120,"walkers":256,"seed":42}"#,
        ),
        (
            "braess_pigou_pricing_challenge",
            r#"{"demand":0.8,"congestion_slope":1.0,"bypass_latency":1.0,"shortcut_cost":0.0,"shortcut_toll":0.5,"shortcut_open":true,"continuum_mfg_mode":false,"mfg_viscosity":0.08,"mfg_grid_points":33}"#,
        ),
        (
            "simpson_deconfounding_challenge",
            r#"{"randomization_type":"Stratified","n_subjects":24,"block_size":4,"simpson_config":{"mild_treatment_share":0.5,"severe_treatment_share":0.5,"stratum_size":400,"mild_counts":{"treat_event":20,"treat_no_event":180,"ctrl_event":30,"ctrl_no_event":170},"severe_counts":{"treat_event":60,"treat_no_event":140,"ctrl_event":80,"ctrl_no_event":120}}}"#,
        ),
    ] {
        let ch = find_goal_challenge(cid);
        let mut fw = SimulationFramework::new(ch.domain);
        fw.launch_challenge(ch).unwrap();
        assert_eq!(fw.evaluate_challenge_predicates(None), Some(false));
        fw.active_tool
            .as_mut()
            .unwrap()
            .try_load_state(solved_state)
            .unwrap();
        assert_eq!(fw.evaluate_challenge_predicates(None), Some(true));
    }
}

#[test]
fn test_braess_and_simpson_spoiler_buttons_gated_during_challenges() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut braess_fw = SimulationFramework::new("paradox_lab");
    braess_fw.select_tool_by_name("BraessNetworkParadoxTool");
    assert!(collect_button_labels(&mut braess_fw, "paradox_lab")
        .iter()
        .any(|l| l.contains("Compare Pigou Toll Preset")));
    braess_fw
        .launch_challenge(find_goal_challenge("braess_pigou_pricing_challenge"))
        .unwrap();
    assert!(!collect_button_labels(&mut braess_fw, "paradox_lab")
        .iter()
        .any(|l| l.contains("Compare Pigou Toll Preset")));

    let mut simpson_fw = SimulationFramework::new("clinical_trials");
    let _ = RandomizationTool::default();
    simpson_fw
        .launch_story(
            inventory::iter::<PresetStoryMetadata>
                .into_iter()
                .find(|s| s.id == "simpson_confounding_reversal_tour")
                .unwrap(),
            0,
        )
        .unwrap();
    assert!(collect_button_labels(&mut simpson_fw, "clinical_trials")
        .iter()
        .any(|l| l.contains("Compare Stratified Balance Preset")));
    simpson_fw
        .launch_challenge(find_goal_challenge("simpson_deconfounding_challenge"))
        .unwrap();
    assert!(!collect_button_labels(&mut simpson_fw, "clinical_trials")
        .iter()
        .any(|l| l.contains("Compare Stratified Balance Preset")));
}

#[test]
fn test_live_ast_sandbox_custom_bounds_and_controller_teleport_clamping() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut braess_fw = SimulationFramework::new("paradox_lab");
    braess_fw
        .launch_challenge(find_goal_challenge("braess_pigou_pricing_challenge"))
        .unwrap();
    braess_fw.open_ast_sandbox("price_of_anarchy > 1.10 && wardrop_latency > closed_latency");
    let _ = braess_fw.evaluate_challenge_predicates(None);
    assert_eq!(braess_fw.challenge_hud.sandbox.last_eval, Ok(true));
    braess_fw.open_ast_sandbox("price_of_anarchy <= 1.005");
    let _ = braess_fw.evaluate_challenge_predicates(None);
    assert_eq!(braess_fw.challenge_hud.sandbox.last_eval, Ok(false));

    let mut simpson_fw = SimulationFramework::new("clinical_trials");
    simpson_fw
        .launch_challenge(find_goal_challenge("simpson_deconfounding_challenge"))
        .unwrap();
    simpson_fw.open_ast_sandbox("pooled_rr > 1.15 && mild_rr < 0.80");
    let _ = simpson_fw.evaluate_challenge_predicates(None);
    assert_eq!(simpson_fw.challenge_hud.sandbox.last_eval, Ok(true));

    let ctx = egui::Context::default();
    let mut kelly = KellyParrondoTool::default();
    let _ = ctx.run(egui::RawInput::default(), |egui_ctx| {
        egui::CentralPanel::default().show(egui_ctx, |ui| kelly.show_inside(ui));
    });
    let low = kelly.atlas.teleport_to(Some(&ctx), -99.0, -99.0).unwrap();
    let high = kelly.atlas.teleport_to(Some(&ctx), 99.0, 99.0).unwrap();
    assert!((low.x_val - 0.35).abs() < 1e-9 && (low.y_val - (-0.02)).abs() < 1e-9);
    assert!((high.x_val - 0.65).abs() < 1e-9 && (high.y_val - 0.045).abs() < 1e-9);
}

// [cite:paradox_lab]
