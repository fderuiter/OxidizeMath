#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::black_hole_geodesic_tool::{BlackHoleGeodesicWorkbench, GeodesicWorkbenchMode};
use super::galactic_orbits_tool::GalacticOrbitSimulator;
use super::relativistic_jet_tool::RelativisticJetBeamingLab;
use super::rotation_curve_tool::RotationCurveExplorer;
use super::*;
use crate::framework::{FiniteStateGuard, InteractiveTool};
use math_explorer::physics::astrophysics::DwarfMorphology;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[test]
fn test_astrophysics_tab_domain_slug_and_tools() {
    let mut tab = AstrophysicsTab::default();
    assert_eq!(tab.name(), "Astrophysics");
    assert_eq!(tab.academic_domain(), "Physics");
    assert_eq!(tab.url_slug(), "astrophysics");
    for id in [
        "RotationCurveExplorer",
        "GalacticOrbitSimulator",
        "BlackHoleGeodesicWorkbench",
        "RelativisticJetBeamingLab",
    ] {
        assert!(tab.select_tool_by_name(id), "missing tool {id}");
        assert_eq!(tab.active_tool_id(), Some(id));
    }
}

#[test]
fn test_type_code_95_to_99_renders_unestablished_badge_without_panic() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = RotationCurveExplorer::default();
    tool.cfg.morphology = DwarfMorphology::TypeCode95To99;
    tool.cfg.use_absolute_magnitude = false;
    tool.recalculate();
    assert!(tool
        .current_baryons()
        .scaling
        .apparent_magnitude_b
        .is_none());

    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| tool.show(ctx));
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit output");
    let found_badge = ak.nodes.iter().any(|(_, n)| {
        n.label()
            .or_else(|| n.value())
            .is_some_and(|l| l.contains("Unestablished in Tillaboev (2025)"))
    });
    assert!(
        found_badge,
        "Expected explicit unestablished-formula badge for TypeCode95To99"
    );
}

#[test]
fn test_galactic_orbits_verified_zero_allocation_and_state_guard() {
    let mut sim = GalacticOrbitSimulator::default();
    let t0 = sim.system.elapsed_myr();
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (calls, ops) = sim.step_inside_verified_engine();
        assert!(calls > 0);
        assert!(ops > 100);
    }
    assert!(sim.system.elapsed_myr() > t0);
    assert!(sim.system.relative_energy_drift() < 1.0e-3);

    sim.cfg.dt_myr = f64::NAN;
    assert!(!sim.has_finite_parameters());
    FiniteStateGuard::clamp_parameters(&mut sim);
    assert!(sim.has_finite_parameters());
    assert_eq!(sim.cfg.halo_kind, HaloKindChoice::BurkertCore);
}

#[test]
fn test_black_hole_workbench_64_ray_verified_zero_alloc_and_benchmarks() {
    let mut wb = BlackHoleGeodesicWorkbench::default();
    let defl = wb
        .bundle
        .probe_photon_deflection_rad()
        .expect("default b = 2.60 R_s scatters");
    assert!((defl - 6.51).abs() < 0.03, "expected ~6.51 rad, got {defl}");

    wb.cfg.mode = GeodesicWorkbenchMode::IscoPlungeComparison;
    wb.rebuild_bundle(800);
    let tr = &wb.bundle.trajectories()[0];
    assert!(tr.halted_horizon);
    assert!((tr.affine_elapsed - 37.03).abs() < 0.25);
    assert!((wb.bundle.newtonian_state().min_radius - 2.883).abs() < 0.02);

    wb.cfg.mode = GeodesicWorkbenchMode::MixedEnsemble64;
    wb.cfg.ray_count = 64;
    wb.rebuild_bundle(0);
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (calls, ops) = wb.step_inside_verified_engine();
        assert!(calls > 0 && ops > 100);
    }
    assert!(wb.bundle.max_killing_drift() < 1.0e-6);

    wb.cfg.d_lambda = f64::NAN;
    assert!(!wb.has_finite_parameters());
    FiniteStateGuard::clamp_parameters(&mut wb);
    assert!(wb.has_finite_parameters());
}

#[test]
fn test_relativistic_jet_lab_verified_zero_alloc_and_beaming_benchmarks() {
    let mut lab = RelativisticJetBeamingLab::default();
    assert!((lab.observables.jet_flux_asymmetry - 18_855.15).abs() < 0.1);
    assert!((lab.observables.disk_flux_asymmetry - 58.257).abs() < 0.05);
    assert!((lab.observables.shock_compression - 7.0).abs() < 1e-12);
    assert!(lab.observables.li_ma_sigma > 5.0);

    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (calls, ops) = lab.recalculate_inside_verified_engine();
        assert!(calls > 0 && ops > 100);
    }

    lab.cfg.bulk_beta = f64::INFINITY;
    assert!(!lab.has_finite_parameters());
    FiniteStateGuard::clamp_parameters(&mut lab);
    assert!(lab.has_finite_parameters());

    assert!(lab.cfg.snapshot_manager.add_snapshot(
        "Pinned jet",
        "beta=0.90",
        vec![crate::framework::TraceSeries::new(
            "lobe",
            vec![[0.0, 0.0], [1.0, 1.0]]
        )],
    ));
    assert!(lab.save_state().unwrap().contains("snapshot_manager"));
    assert!(!lab.save_url_state().unwrap().contains("snapshot_manager"));
}

#[test]
fn test_astrophysics_mystery_halo_fit_zero_alloc_residuals_and_clamping() {
    use super::challenge::{
        CHALLENGE_ASTRO_MYSTERY_HALO_FIT, LOCKED_DISTANCE_MPC, MYSTERY_DWARF_TARGETS,
    };
    use crate::starter_gallery::find_challenge_metadata;

    assert_eq!(MYSTERY_DWARF_TARGETS.len(), 16);
    let meta = find_challenge_metadata(CHALLENGE_ASTRO_MYSTERY_HALO_FIT).expect("registered");
    let baseline = meta.generate_baseline_state();
    let mut tool = RotationCurveExplorer::default();
    tool.apply_challenge(Some(CHALLENGE_ASTRO_MYSTERY_HALO_FIT))
        .expect("challenge applies");
    tool.try_load_state(&baseline).expect("baseline loads");

    let base_eval = tool.evaluate_challenge().expect("eval");
    assert!(!base_eval.passed, "Baseline NFW halo must fail fit");
    assert!(tool.rmse_kms > 10.0 && tool.reduced_chi_sq > 10.0);

    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            tool.cfg.halo_kind = HaloKindChoice::BurkertCore;
            tool.cfg.halo_log10_mass = 10.15;
            tool.cfg.halo_scale_kpc = 2.40;
            tool.cfg.halo_concentration = 13.0;
            tool.recalculate_zero_alloc();
        });
    }

    let solved_eval = tool.evaluate_challenge().expect("eval");
    assert!(
        solved_eval.passed,
        "Cored Burkert profile must pass mystery dwarf fit: {solved_eval:?}"
    );
    assert!(tool.rmse_kms <= 2.40 && tool.reduced_chi_sq <= 1.35);
    assert!(
        tool.counterpart_reduced_chi_sq > 5.0,
        "NFW counterpart at identical parameters must fail cored inner slope"
    );

    tool.sync_rotation_callouts();
    assert!(tool.callout_manager.callouts().len() >= 2);

    let tampered = r#"{"distance_mpc":88.0,"abs_magnitude_v":-19.0,"use_absolute_magnitude":true,"morphology":"GeneralIrregular","gas_fraction":0.10,"disk_scale_kpc":3.0,"halo_kind":"BurkertCore","halo_log10_mass":10.15,"halo_scale_kpc":2.40,"halo_concentration":13.0,"max_radius_kpc":22.0,"compare_counterpart":true}"#;
    let clamped = meta.clamp_state_json(tampered).expect("clamps cleanly");
    let parsed: super::rotation_curve_tool::RotationCurveState =
        serde_json::from_str(&clamped).unwrap();
    assert!((parsed.distance_mpc - LOCKED_DISTANCE_MPC).abs() < 1e-9);
    assert!(!parsed.use_absolute_magnitude);
    assert_eq!(parsed.morphology, DwarfMorphology::TypeCode10);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn wasm_astrophysics_tab_url_roundtrip_and_symplectic_stepping() {
    use super::challenge::CHALLENGE_ASTRO_MYSTERY_HALO_FIT;
    use crate::starter_gallery::find_challenge_metadata;

    let mut tab = AstrophysicsTab::default();
    assert_eq!(tab.url_slug(), "astrophysics");
    assert_eq!(tab.academic_domain(), "Physics");

    assert!(tab.select_tool_by_name("RotationCurveExplorer"));
    let mut rc = RotationCurveExplorer::default();
    rc.cfg.morphology = DwarfMorphology::TypeCode95To99;
    rc.cfg.halo_kind = HaloKindChoice::NfwCusp;
    rc.cfg.distance_mpc = 19.5;
    rc.recalculate();
    assert!(rc.current_baryons().scaling.apparent_magnitude_b.is_none());
    let exported = rc.export_preset_json().expect("export preset json");
    tab.import_url_preset_json(Some("RotationCurveExplorer"), Some(&exported))
        .expect("import url preset");
    assert!(tab.export_url_preset_json().unwrap().is_some());

    let meta = find_challenge_metadata(CHALLENGE_ASTRO_MYSTERY_HALO_FIT).expect("registered");
    rc.apply_challenge(Some(CHALLENGE_ASTRO_MYSTERY_HALO_FIT))
        .expect("challenge enabled");
    rc.try_load_state(&meta.generate_baseline_state())
        .expect("baseline loaded");
    assert!(!rc.evaluate_challenge().expect("eval").passed);

    let mut orbits = GalacticOrbitSimulator::default();
    let initial_t = orbits.system.elapsed_myr();
    orbits.step_simulation();
    assert!(orbits.system.elapsed_myr() > initial_t);
    assert!(orbits.has_finite_parameters());
    assert!(!orbits.export_datasets().is_empty());

    let mut bh = BlackHoleGeodesicWorkbench::default();
    bh.step_simulation();
    assert!(bh.has_finite_parameters());
    assert!(!bh.export_datasets().is_empty());

    let jet = RelativisticJetBeamingLab::default();
    let jet_json = jet.export_preset_json().expect("jet preset json");
    tab.import_url_preset_json(Some("RelativisticJetBeamingLab"), Some(&jet_json))
        .expect("import jet preset");
    assert!(!jet.export_datasets().is_empty());
}
