#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{ChallengeHudMode, CompiledGoalPredicate, SimulationFramework};
use crate::starter_gallery::{
    find_challenge_by_id, find_story_by_id, get_story_cards_for_domain,
    show_starter_gallery_overlay, StarterGalleryTab,
};
use crate::web_io::PlatformInstant;
use eframe::egui;
use std::collections::HashMap;

#[test]
fn test_compiled_goal_predicate_ast_eval_depth_finite_and_latency() {
    let pred = CompiledGoalPredicate::compile(
        "expected_value > 0.05 && stake_multiplier > 2.0 && expected_growth_rate < -0.01",
    )
    .expect("valid predicate");
    let mut scratch = HashMap::new();
    let unsolved = [
        ("expected_value", 0.20),
        ("stake_multiplier", 1.0),
        ("expected_growth_rate", 0.0201),
    ];
    let solved = [
        ("expected_value", 0.20),
        ("stake_multiplier", 2.5),
        ("expected_growth_rate", -0.034),
    ];
    assert!(!pred.eval_observables(&unsolved, &mut scratch).unwrap());
    assert!(pred.eval_observables(&solved, &mut scratch).unwrap());

    let nan_obs = [
        ("expected_value", f64::NAN),
        ("stake_multiplier", 2.5),
        ("expected_growth_rate", -0.034),
    ];
    assert!(!pred.eval_observables(&nan_obs, &mut scratch).unwrap());

    let deep_nested = format!("{}x{} > 0.0", "sin(".repeat(70), ")".repeat(70));
    assert!(CompiledGoalPredicate::compile(&deep_nested).is_err());

    let bounded_nested = format!("{}x{} > -2.0", "sin(".repeat(32), ")".repeat(32));
    let bounded_pred = CompiledGoalPredicate::compile(&bounded_nested).expect("parses depth 32");
    assert!(bounded_pred
        .eval_observables(&[("x", 1.0)], &mut scratch)
        .unwrap());

    let start = PlatformInstant::now();
    for _ in 0..200 {
        assert!(pred.eval_observables(&solved, &mut scratch).unwrap());
    }
    let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / 200.0;
    assert!(
        avg_ms < 0.2,
        "AST predicate evaluation averaged {avg_ms:.4} ms (budget < 0.2 ms)"
    );
}

#[test]
fn test_kelly_challenge_completes_and_announces_once_via_live_region() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let challenge =
        find_challenge_by_id("kelly_ruin_trap", Some("financial_math")).expect("kelly challenge");
    let mut fw = SimulationFramework::new("financial_math");
    fw.launch_challenge(challenge).expect("launches challenge");
    assert_eq!(fw.evaluate_challenge_predicates(Some(&ctx)), Some(false));

    let overbet_state = r#"{"initial_bankroll":1000.0,"probability":0.6,"odds":1.0,"num_bets":200,"stake_multiplier":2.5}"#;
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(overbet_state)
        .expect("loads overbet state");

    ctx.data_mut(|d| d.remove::<bool>(egui::Id::new("aria_live_dispatched")));
    assert_eq!(fw.evaluate_challenge_predicates(Some(&ctx)), Some(true));
    let msg: Option<String> = ctx.data(|d| d.get_temp(egui::Id::new("aria_live_message")));
    assert!(msg
        .as_deref()
        .is_some_and(|m| m.contains("Kelly Ruin Trap")));

    ctx.data_mut(|d| d.remove::<bool>(egui::Id::new("aria_live_dispatched")));
    assert_eq!(fw.evaluate_challenge_predicates(Some(&ctx)), Some(true));
    let dispatched: Option<bool> = ctx.data(|d| d.get_temp(egui::Id::new("aria_live_dispatched")));
    assert_eq!(dispatched, None);
}

#[test]
fn test_story_transitions_announce_each_step_once_and_pause_on_diverged() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let story =
        find_story_by_id("dalembert_to_kutta_lift", Some("fluid_dynamics")).expect("lift story");
    let mut fw = SimulationFramework::new("fluid_dynamics");
    fw.launch_story(story, 0).expect("launches step 0");

    ctx.data_mut(|d| d.remove::<bool>(egui::Id::new("aria_live_dispatched")));
    fw.flush_hud_announcements(&ctx);
    let first_msg: Option<String> = ctx.data(|d| d.get_temp(egui::Id::new("aria_live_message")));
    assert!(first_msg
        .as_deref()
        .is_some_and(|m| m.contains("step 1 of 3")));

    ctx.data_mut(|d| d.remove::<bool>(egui::Id::new("aria_live_dispatched")));
    fw.flush_hud_announcements(&ctx);
    assert_eq!(
        ctx.data(|d| d.get_temp::<bool>(egui::Id::new("aria_live_dispatched"))),
        None
    );

    fw.step_story(1).expect("advances to step 1");
    fw.flush_hud_announcements(&ctx);
    let second_msg: Option<String> = ctx.data(|d| d.get_temp(egui::Id::new("aria_live_message")));
    assert!(second_msg
        .as_deref()
        .is_some_and(|m| m.contains("step 2 of 3")));

    let challenge =
        find_challenge_by_id("kutta_joukowski_lift", Some("fluid_dynamics")).expect("challenge");
    fw.launch_challenge(challenge)
        .expect("launches lift challenge");
    let baseline = fw.baseline_tool_state.clone();
    fw.is_diverged = true;
    assert_eq!(fw.evaluate_challenge_predicates(Some(&ctx)), None);
    fw.reset_parameters();
    assert!(!fw.is_diverged);
    assert_eq!(
        fw.active_tool.as_ref().and_then(|t| t.save_state()),
        baseline
    );
    assert!(matches!(
        fw.challenge_hud.mode,
        ChallengeHudMode::Challenge { .. }
    ));
}

#[test]
fn test_url_hydration_for_challenge_and_story_and_unknown_id_warning_toast() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = crate::app::MathExplorerApp::default();
    let ctx = egui::Context::default();
    let ok = app
        .apply_share_location_strings(
            "#tab=financial_math&tool=BankrollGrowthTool&challenge=kelly_ruin_trap",
            "",
            Some(&ctx),
        )
        .expect("valid challenge URL hydrates");
    assert!(ok);
    assert!(app.session_manager.toast().is_none());
    let tab = app.tabs[app.selected_tab]
        .framework()
        .expect("framework tab");
    assert_eq!(tab.active_tool_id(), Some("BankrollGrowthTool"));
    assert!(matches!(
        tab.challenge_hud.mode,
        ChallengeHudMode::Challenge { challenge, .. } if challenge.id == "kelly_ruin_trap"
    ));

    let encoded = tab
        .encode_share_fragment_for_tab("financial_math")
        .expect("encodes compact challenge URL");
    assert_eq!(
        encoded,
        "#tab=financial_math&tool=BankrollGrowthTool&challenge=kelly_ruin_trap"
    );

    let bad = app.apply_share_location_strings(
        "#tab=financial_math&tool=BankrollGrowthTool&challenge=nonexistent_id",
        "",
        Some(&ctx),
    );
    assert!(bad.is_err());
    assert!(app
        .session_manager
        .toast()
        .is_some_and(|(msg, is_err)| *is_err && msg.contains("nonexistent_id")));
}

#[test]
fn test_starter_gallery_filter_tabs_render_stories_and_challenges() {
    let ctx = egui::Context::default();
    let screen_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let cards = get_story_cards_for_domain("game_theory");
    let mut open = true;
    for tab_mode in [
        StarterGalleryTab::Snapshots,
        StarterGalleryTab::GuidedStories,
        StarterGalleryTab::Challenges,
    ] {
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen_rect),
                ..Default::default()
            },
            |egui_ctx| {
                egui_ctx.data_mut(|d| {
                    d.insert_temp(
                        egui::Id::new(("game_theory", "starter_gallery_active_tab")),
                        tab_mode,
                    );
                });
                let _ = show_starter_gallery_overlay(egui_ctx, "game_theory", &cards, &mut open);
            },
        );
    }
    assert!(open);
}
