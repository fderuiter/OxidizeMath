//! Acceptance tests for `LeniaSpectralCartographer`, live `U = A * K` potential overlays,
//! `CompiledExpr` growth/kernel sandboxes, URL round-tripping, and Lenia challenges/stories.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::atlas::{
    LeniaRegime, LeniaSpectralCartographer, WASHOUT_MASS_THRESHOLD,
    gaussian_vacuum_boundary_slope,
};
use super::challenge::CHALLENGE_ORBIUM_TAIL_REGEN;
use super::sandbox::{CHAN_POLYNOMIAL_GROWTH_EXPR, CHAN_POLYNOMIAL_KERNEL_EXPR};
use super::*;
use crate::async_sim::atlas::RegimeCartographer;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::starter_gallery::{GoalChallengeMetadata, PresetStoryMetadata, find_challenge_metadata};
use crate::url_state::{MAX_DECODED_URL_STATE_BYTES, encode_base64url};

fn screen_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

#[test]
fn test_atlas_teleport_auto_reseeds_extinct_torus_and_sustains_orbium_80_steps() {
    let ctx = egui::Context::default();
    let mut tool = LeniaTool {
        mu: 0.35,
        sigma: 0.012,
        ..Default::default()
    };

    // Drive torus into Extinction (total_mass < 0.25)
    tool.engine.clear();
    tool.record_telemetry_sample();
    assert!(tool.engine.telemetry().total_mass < WASHOUT_MASS_THRESHOLD);

    // Open (mu, sigma) atlas and render a frame so active_cartographer is populated
    tool.atlas.set_visible(true);
    let _ = ctx.run(screen_input(), |ctx| tool.show(ctx));

    let ev = tool
        .atlas
        .teleport_to(Some(&ctx), 0.15, 0.015)
        .expect("teleport inside (mu, sigma) atlas");
    assert_eq!(ev.regime_label, LeniaRegime::LocalizedSoliton.label());

    tool.apply_atlas_teleport(ev);
    assert!((tool.mu - 0.15).abs() < 1e-12);
    assert!((tool.sigma - 0.015).abs() < 1e-12);
    assert!(
        tool.engine.telemetry().total_mass > 20.0,
        "Extinct torus must auto-reseed Orbium on Localized Soliton teleport"
    );

    let cx0 = tool.engine.telemetry().centroid_x;
    let cy0 = tool.engine.telemetry().centroid_y;
    for _ in 0..80 {
        tool.step_once();
    }
    let tel = tool.engine.telemetry();
    let disp = (tel.centroid_x - cx0).hypot(tel.centroid_y - cy0);
    assert!(
        disp > 3.5 && tel.centroid_speed > 0.05 && tel.total_mass > 20.0,
        "Reseeded Orbium must sustain translation over 80 steps: disp={disp}, tel={tel:?}"
    );

    // Verify ("radius", "mu") secondary axis pair also resolves and evaluates cleanly
    tool.atlas
        .set_axis_keys("radius", "mu")
        .expect("radius vs mu axis pair");
    let cart_rm = LeniaSpectralCartographer::from_tool(&tool, ("radius", "mu"));
    let cell = cart_rm.evaluate_cell(13.0, 0.15);
    assert_eq!(
        LeniaRegime::from_u8(cell.regime),
        LeniaRegime::LocalizedSoliton
    );
}

#[test]
fn test_cartographer_overlay_contour_and_zero_heap_allocations() {
    let mut tool = LeniaTool::default();
    let gauss_cart = LeniaSpectralCartographer::from_tool(&tool, ("mu", "sigma"));
    let (label, pts) = gauss_cart
        .overlay_contour()
        .expect("Gaussian G(0) = 0 contour");
    assert!(label.contains("G(0) = 0"));
    assert!(pts.len() >= 16);
    let slope = gaussian_vacuum_boundary_slope();
    for &[mu, sigma] in &pts {
        assert!(
            (mu - slope * sigma).abs() < 1e-12,
            "Contour point ({mu}, {sigma}) must match mu = sqrt(2 ln 2) * sigma"
        );
    }

    tool.set_growth_expression(CHAN_POLYNOMOM_TEST_EXPR);
    tool.set_kernel_expression(CHAN_POLYNOMIAL_KERNEL_EXPR);
    let custom_cart = LeniaSpectralCartographer::from_tool(&tool, ("mu", "sigma"));
    assert!(custom_cart.overlay_contour().is_some());

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((c_loc, c_vac, c_custom), _) =
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            (
                gauss_cart.evaluate_cell(0.15, 0.015),
                gauss_cart.evaluate_cell(0.02, 0.05),
                custom_cart.evaluate_cell(0.15, 0.015),
            )
        });
    assert_eq!(
        LeniaRegime::from_u8(c_loc.regime),
        LeniaRegime::LocalizedSoliton
    );
    assert_eq!(
        LeniaRegime::from_u8(c_vac.regime),
        LeniaRegime::VacuumExplosion
    );
    assert_eq!(
        LeniaRegime::from_u8(c_custom.regime),
        LeniaRegime::LocalizedSoliton
    );
}

const CHAN_POLYNOMOM_TEST_EXPR: &str = CHAN_POLYNOMIAL_GROWTH_EXPR;

#[test]
fn test_potential_histogram_overlay_and_exported_datasets() {
    let ctx = egui::Context::default();
    let mut tool = LeniaTool::default();
    let _ = ctx.run(screen_input(), |ctx| tool.show(ctx));

    let dist = compute_potential_distribution(tool.engine.state(), tool.engine.potential());
    assert!(dist.active_count > 50);
    assert!(
        (dist.u_max - 0.225).abs() < 0.035,
        "Orbium max(U)={} expected near 0.225",
        dist.u_max
    );
    assert!(dist.u_p05 > 0.0 && dist.u_p05 < dist.u_max);

    let datasets = tool.export_datasets();
    let hist = datasets
        .iter()
        .find(|(name, _)| name == "Potential Histogram U = A * K")
        .expect("Potential histogram series in exported datasets");
    assert_eq!(hist.1.len(), 32);
    assert!(hist.1.iter().any(|p| p[1] > 0.5));

    let bounds = datasets
        .iter()
        .find(|(name, _)| name == "Potential Niche Bounds [u_p05, max(U)]")
        .expect("Potential niche bounds series in exported datasets");
    assert_eq!(bounds.1.len(), 2);
    assert!((bounds.1[1][0] - dist.u_max).abs() < 1e-12);
}

#[test]
fn test_chan_polynomial_sandbox_synthesizes_sliders_roundtrips_url_and_sustains_orbium() {
    let mut framework = SimulationFramework::new("biology");
    assert!(framework.select_tool_by_name("LeniaTool"));
    {
        let tool = framework
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<LeniaTool>())
            .expect("LeniaTool active");
        tool.set_growth_expression("2 * max(0, 1 - ((u - mu) / (3 * sigma))^2)^4 - 1");
        assert_eq!(tool.growth_bar.active_param_names(), &["mu", "sigma"]);
        assert!((tool.growth_bar.active_param_value("mu").unwrap() - 0.15).abs() < 1e-12);
        assert!((tool.growth_bar.active_param_value("sigma").unwrap() - 0.015).abs() < 1e-12);

        // Verify transient invalid edit keeps last_valid_compiled without FiniteStateGuard divergence
        tool.growth_bar.set_expression("2 * max(0, ");
        tool.step_once();
        assert!(FiniteStateGuard::is_finite_state(tool));

        // Restore valid Chan polynomial before URL roundtrip
        tool.set_growth_expression("2 * max(0, 1 - ((u - mu) / (3 * sigma))^2)^4 - 1");
    }

    let state_json = framework
        .export_url_preset_json()
        .expect("export json")
        .expect("some json");
    assert!(state_json.len() < MAX_DECODED_URL_STATE_BYTES);
    assert!(state_json.contains("growth_expr"));

    let share_url = framework.encode_share_url().expect("encode share url");
    let hash_idx = share_url.find('#').expect("hash");
    let fragment = &share_url[hash_idx..];

    let mut app = crate::app::MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings(fragment, "", None)
        .expect("hydrate share URL"));
    let bio_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .unwrap();
    assert_eq!(app.tabs[bio_idx].active_tool_id(), Some("LeniaTool"));
    assert_eq!(
        app.tabs[bio_idx].export_url_preset_json().unwrap(),
        Some(state_json)
    );

    // Step hydrated Chan polynomial Orbium 80 steps and verify viability
    let mut check_tool = LeniaTool::default();
    check_tool.set_growth_expression("2 * max(0, 1 - ((u - mu) / (3 * sigma))^2)^4 - 1");
    for _ in 0..80 {
        check_tool.step_once();
    }
    let tel = check_tool.engine.telemetry();
    assert!(
        tel.total_mass > 20.0 && tel.centroid_speed > 0.05,
        "Chan polynomial Orbium must remain viable: {tel:?}"
    );
}

fn assert_tampered_challenge_url_clamps_radius(meta: &crate::framework::ChallengeMetadata) {
    let mut tampered: serde_json::Value =
        serde_json::from_str(&meta.generate_baseline_state()).unwrap();
    tampered["radius"] = serde_json::json!(28.0);
    tampered["mu"] = serde_json::json!(0.15);
    tampered["sigma"] = serde_json::json!(0.015);
    let encoded = encode_base64url(tampered.to_string().as_bytes());
    let fragment = format!(
        "#tab=Morphogenesis&tool=LeniaTool&challenge={CHALLENGE_ORBIUM_TAIL_REGEN}&state={encoded}"
    );

    let mut app = crate::app::MathExplorerApp::default();
    assert!(
        app.apply_share_location_strings(&fragment, "", None)
            .expect("hydrate challenge URL")
    );
    let bio_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .unwrap();
    let hydrated_json = app.tabs[bio_idx]
        .export_url_preset_json()
        .unwrap()
        .expect("hydrated challenge json");
    let hydrated_env: serde_json::Value = serde_json::from_str(&hydrated_json).unwrap();
    let hydrated_state: LeniaToolState =
        serde_json::from_value(hydrated_env["config"].clone()).unwrap();
    assert!(
        (hydrated_state.radius - 13.0).abs() < 1e-12,
        "Tampered radius must be clamped to 13.0, got {}",
        hydrated_state.radius
    );
}

#[test]
fn test_orbium_tail_regeneration_challenge_url_clamp_and_healing_pass() {
    let meta = find_challenge_metadata(CHALLENGE_ORBIUM_TAIL_REGEN)
        .expect("lenia_orbium_tail_regeneration registered");
    assert_eq!(meta.domain, "biology");

    // Intact 64x64 Orbium reference mass vs tail-cut challenge mass
    let mut intact = LeniaTool {
        grid_size: 64,
        ..Default::default()
    };
    intact.reseed_current_genome();
    let intact_mass = intact.engine.telemetry().total_mass;

    let mut fw = SimulationFramework::new("biology");
    fw.launch_challenge_by_id(CHALLENGE_ORBIUM_TAIL_REGEN, None)
        .expect("launch tail-regeneration challenge");
    assert_eq!(fw.active_challenge_id(), Some(CHALLENGE_ORBIUM_TAIL_REGEN));

    {
        let tool = fw
            .active_tool
            .as_ref()
            .and_then(|t| t.as_any())
            .and_then(|a| a.downcast_ref::<LeniaTool>())
            .unwrap();
        assert!(
            tool.engine.telemetry().total_mass < intact_mass - 3.0,
            "Tail-cut Orbium mass {} must be lower than intact {intact_mass}",
            tool.engine.telemetry().total_mass
        );
        assert!(!tool.evaluate_challenge().unwrap().passed);
    }

    assert_tampered_challenge_url_clamps_radius(meta);

    // Tune (mu, sigma) to (0.15, 0.015), step until healed mass and speed stabilize, and verify pass
    let tool = fw
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut())
        .and_then(|a| a.downcast_mut::<LeniaTool>())
        .unwrap();
    tool.mu = 0.15;
    tool.sigma = 0.015;
    tool.reseed_current_genome();
    for _ in 0..60 {
        tool.step_once();
    }
    let eval = tool.evaluate_challenge().expect("challenge evaluation");
    assert!(
        eval.passed && (eval.score - 100.0).abs() < 1e-9,
        "Healed Orbium must pass tail-regeneration challenge: {eval:?}"
    );
}

#[test]
fn test_lenia_story_tour_and_geminium_fission_goal_challenge() {
    let story = inventory::iter::<PresetStoryMetadata>
        .into_iter()
        .find(|s| s.id == "lenia_soliton_viability_tour")
        .expect("lenia_soliton_viability_tour registered");
    let mut fw = SimulationFramework::new("biology");
    fw.launch_story(story, 0).expect("launch step 0");
    let tool = fw
        .active_tool
        .as_ref()
        .and_then(|t| t.as_any())
        .and_then(|a| a.downcast_ref::<LeniaTool>())
        .unwrap();
    assert!(!tool.callouts.is_empty());

    let goal = inventory::iter::<GoalChallengeMetadata>
        .into_iter()
        .find(|g| g.id == "lenia_geminium_fission")
        .expect("lenia_geminium_fission registered");
    fw.launch_challenge(goal).expect("launch geminium goal");
    assert_eq!(fw.evaluate_challenge_predicates(None), Some(false));

    {
        let tool_mut = fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<LeniaTool>())
            .unwrap();
        tool_mut.mu = 0.24;
        tool_mut.sigma = 0.036;
        for _ in 0..45 {
            tool_mut.step_once();
        }
    }
    let obs = fw.active_tool.as_ref().unwrap().observables();
    assert_eq!(
        fw.evaluate_challenge_predicates(None),
        Some(true),
        "Geminium tuned to (0.24, 0.036) must satisfy fission goal predicate; obs={obs:?}"
    );
}
