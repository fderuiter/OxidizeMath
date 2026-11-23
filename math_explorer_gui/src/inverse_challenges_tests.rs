//! Integration tests for `ChallengeMetadata`, `StarterGallery` Challenges tab, Challenge HUD,
//! WCAG 2.2 AA contrast, `CalloutManager` >= 5 tools adoption, and stateless challenge URLs.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::app::{MathExplorerApp, SemanticPalette, ThemeMode};
use crate::framework::{ChallengeMetadata, SimulationFramework};
use crate::starter_gallery::{
    find_challenge_metadata, get_challenge_cards_for_domain, show_starter_gallery_with_challenges,
};
use crate::url_state::{
    encode_base64url, parse_share_location, MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES,
};
use eframe::egui::{self, Color32};

fn srgb_channel_to_linear(c: u8) -> f64 {
    let v = f64::from(c) / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn relative_luminance(c: Color32) -> f64 {
    0.2126 * srgb_channel_to_linear(c.r())
        + 0.7152 * srgb_channel_to_linear(c.g())
        + 0.0722 * srgb_channel_to_linear(c.b())
}

fn contrast_ratio(fg: Color32, bg: Color32) -> f64 {
    let l1 = relative_luminance(fg);
    let l2 = relative_luminance(bg);
    let (lighter, darker) = if l1 >= l2 { (l1, l2) } else { (l2, l1) };
    (lighter + 0.05) / (darker + 0.05)
}

#[test]
fn test_federated_challenge_registry_and_gallery_tab_switch() {
    let cards: Vec<_> = inventory::iter::<ChallengeMetadata>
        .into_iter()
        .map(ChallengeMetadata::to_card)
        .collect();
    assert!(
        cards.len() >= 5,
        "Expected at least 5 registered ChallengeMetadata entries, got {}",
        cards.len()
    );
    for required_id in [
        "bloch_pi_inversion",
        "bloch_hahn_spin_echo",
        "pk_therapeutic_window",
        "lbm_low_drag_fairing",
        "imrt_oar_sparing",
    ] {
        let meta = find_challenge_metadata(required_id)
            .unwrap_or_else(|| panic!("Missing ChallengeMetadata '{required_id}'"));
        let baseline = meta.generate_baseline_state();
        let clamped = meta
            .clamp_state_json(&baseline)
            .expect("baseline state must pass clamp_state_json");
        assert!(clamped.len() <= MAX_DECODED_URL_STATE_BYTES);
    }

    assert_eq!(get_challenge_cards_for_domain("mri").len(), 2);
    assert_eq!(get_challenge_cards_for_domain("clinical_trials").len(), 1);
    assert_eq!(get_challenge_cards_for_domain("fluid_dynamics").len(), 1);
    assert_eq!(get_challenge_cards_for_domain("medical").len(), 1);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut open = true;
    let mri_challenges = get_challenge_cards_for_domain("mri");
    let out = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = show_starter_gallery_with_challenges(ctx, "mri", &[], &mri_challenges, &mut open);
    });
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit nodes expected");
    assert!(ak.nodes.iter().any(|(_, n)| n
        .label()
        .unwrap_or("")
        .contains("Bloch 180° Spin Inversion")));
}

fn assert_theme_hud_contrast(theme: ThemeMode, ctx: &egui::Context) {
    let visuals = ctx.style().visuals.clone();
    let palette = SemanticPalette::from_visuals(&visuals);
    let bg = visuals.window_fill;
    for token in [
        palette.info,
        palette.success,
        palette.warning,
        palette.error,
        palette.muted_text,
        visuals.text_color(),
    ] {
        let ratio = contrast_ratio(token, bg);
        assert!(
            ratio >= 4.5,
            "Theme {theme:?} HUD text token contrast {ratio:.2}:1 < 4.5:1"
        );
    }
    for stroke_token in [palette.graph_edge, palette.success, palette.info] {
        let ratio = contrast_ratio(stroke_token, bg);
        assert!(
            ratio >= 3.0,
            "Theme {theme:?} HUD stroke contrast {ratio:.2}:1 < 3.0:1"
        );
    }
}

#[test]
fn test_challenge_hud_wcag_contrast_reset_and_aria_across_themes() {
    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let ctx = egui::Context::default();
        theme.apply_to_context(&ctx);
        assert_theme_hud_contrast(theme, &ctx);

        let mut fw = SimulationFramework::new("clinical_trials");
        fw.launch_challenge_by_id("pk_therapeutic_window", Some(&ctx))
            .expect("pk_therapeutic_window must launch");
        assert_eq!(fw.active_challenge_id(), Some("pk_therapeutic_window"));
        let base_eval = fw
            .active_tool
            .as_ref()
            .and_then(|t| t.evaluate_challenge())
            .unwrap();
        assert!(!base_eval.passed, "Unsolved baseline must not start passed");

        crate::accessibility::clear_test_announcements();
        let mut solved_cfg: serde_json::Value =
            serde_json::from_str(&fw.active_tool.as_ref().unwrap().save_state().unwrap()).unwrap();
        solved_cfg["dose_mg"] = serde_json::json!(165.0);
        solved_cfg["use_loading_dose"] = serde_json::json!(true);
        solved_cfg["loading_dose_mg"] = serde_json::json!(240.0);
        solved_cfg["interval_hours"] = serde_json::json!(6.0);
        fw.active_tool
            .as_mut()
            .unwrap()
            .try_load_state(&solved_cfg.to_string())
            .unwrap();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            fw.show(ctx, "clinical_trials");
        });
        let announcements = crate::accessibility::all_announcements();
        assert!(announcements
            .iter()
            .any(|(msg, prio)| msg.contains("Challenge passed") && prio == "polite"));

        fw.reset_active_challenge_baseline(Some(&ctx)).unwrap();
        assert!(
            !fw.active_tool
                .as_ref()
                .unwrap()
                .evaluate_challenge()
                .unwrap()
                .passed
        );
        fw.exit_active_challenge(Some(&ctx));
        assert!(fw.active_challenge_id().is_none());
    }
}

#[test]
fn test_stateless_challenge_urls_within_size_budget() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for (slug, challenge_id) in [
        ("mri", "bloch_pi_inversion"),
        ("mri", "bloch_hahn_spin_echo"),
        ("clinical_trials", "pk_therapeutic_window"),
        ("fluid_dynamics", "lbm_low_drag_fairing"),
        ("medical", "imrt_oar_sparing"),
    ] {
        let bare_hash = format!("#tab={slug}&challenge={challenge_id}");
        let parsed = parse_share_location(&bare_hash, "").unwrap().unwrap();
        assert_eq!(parsed.challenge.as_deref(), Some(challenge_id));

        let mut app = MathExplorerApp::default();
        assert!(app
            .apply_share_location_strings(&bare_hash, "", None)
            .unwrap());
        let active_fw = app.tabs[app.selected_tab].framework().unwrap();
        assert_eq!(active_fw.active_challenge_id(), Some(challenge_id));

        let full_url = app.encode_current_share_url().unwrap();
        let fragment = &full_url[full_url.find('#').unwrap()..];
        assert!(fragment.len() <= MAX_URL_FRAGMENT_BYTES);
        let full_params = parse_share_location(fragment, "").unwrap().unwrap();
        assert_eq!(full_params.challenge.as_deref(), Some(challenge_id));
        assert!(full_params.state_json.unwrap().len() <= MAX_DECODED_URL_STATE_BYTES);
    }
}

#[test]
fn test_tampered_challenge_url_clamps_locked_parameters() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    app.apply_share_location_strings(
        "#tab=clinical_trials&challenge=pk_therapeutic_window",
        "",
        None,
    )
    .unwrap();
    let url = app.encode_current_share_url().unwrap();
    let frag = &url[url.find('#').unwrap()..];
    let params = parse_share_location(frag, "").unwrap().unwrap();
    let mut envelope: serde_json::Value =
        serde_json::from_str(params.state_json.as_ref().unwrap()).unwrap();
    envelope["config"]["dose_mg"] = serde_json::json!(165.0);
    envelope["config"]["interval_hours"] = serde_json::json!(6.0);
    envelope["config"]["mtc"] = serde_json::json!(95.0);
    envelope["config"]["ke"] = serde_json::json!(0.45);
    let tampered_b64 = encode_base64url(envelope.to_string().as_bytes());
    let tampered_hash = format!(
        "#tab=clinical_trials&tool=PharmacokineticsTool&challenge=pk_therapeutic_window&state={tampered_b64}"
    );

    let mut hydrated_app = MathExplorerApp::default();
    assert!(hydrated_app
        .apply_share_location_strings(&tampered_hash, "", None)
        .unwrap());
    let fw = hydrated_app.tabs[hydrated_app.selected_tab]
        .framework()
        .unwrap();
    assert_eq!(fw.active_challenge_id(), Some("pk_therapeutic_window"));
    let restored_cfg: serde_json::Value =
        serde_json::from_str(&fw.active_tool.as_ref().unwrap().save_state().unwrap()).unwrap();
    assert_eq!(restored_cfg["dose_mg"], serde_json::json!(165.0));
    assert_eq!(restored_cfg["interval_hours"], serde_json::json!(6.0));
    assert_eq!(restored_cfg["mtc"], serde_json::json!(12.0));
    assert_eq!(restored_cfg["ke"], serde_json::json!(0.12));
    let eval = fw
        .active_tool
        .as_ref()
        .unwrap()
        .evaluate_challenge()
        .unwrap();
    assert!(eval.passed && (eval.score - 100.0).abs() < 1e-9);
}

#[test]
#[cfg(all(feature = "pure_math", feature = "applied", feature = "physics"))]
fn test_callout_manager_adopted_by_at_least_five_tools() {
    use crate::async_sim::unified::UnifiedSimTool;
    use crate::framework::{CalloutManager, InteractiveTool};
    use crate::tabs::analysis::ode::OdeSolverTool;
    use crate::tabs::clinical_trials::pharmacokinetics::PharmacokineticsTool;
    use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
    use crate::tabs::medical::dose::DoseCalculationTool;
    use crate::tabs::mri::MriTool;

    fn has_auto(cm: &CalloutManager) -> usize {
        usize::from(cm.callouts().iter().any(|c| c.tag.is_some()))
    }

    let ctx = egui::Context::default();
    let mut ode = OdeSolverTool::default();
    let mut mri = MriTool::default();
    let mut pk = PharmacokineticsTool::default();
    let mut lbm = UnifiedSimTool::<LbmUnified>::new();
    let mut dose = DoseCalculationTool::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        ode.show(ctx);
        mri.show(ctx);
        pk.show(ctx);
        lbm.show(ctx);
        dose.show(ctx);
    });
    let adopted = has_auto(&ode.callout_manager)
        + has_auto(&mri.callout_manager)
        + has_auto(&pk.callout_manager)
        + has_auto(&lbm.callout_manager)
        + has_auto(&dose.callout_manager);
    assert!(adopted >= 5);
}
