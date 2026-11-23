#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::galactic_orbits_tool::GalacticOrbitSimulator;
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
    assert!(tab.select_tool_by_name("RotationCurveExplorer"));
    assert_eq!(tab.active_tool_id(), Some("RotationCurveExplorer"));
    assert!(tab.select_tool_by_name("GalacticOrbitSimulator"));
    assert_eq!(tab.active_tool_id(), Some("GalacticOrbitSimulator"));
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn wasm_astrophysics_tab_url_roundtrip_and_symplectic_stepping() {
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

    let mut orbits = GalacticOrbitSimulator::default();
    let initial_t = orbits.system.elapsed_myr();
    orbits.step_simulation();
    assert!(orbits.system.elapsed_myr() > initial_t);
    assert!(orbits.has_finite_parameters());
    assert!(!orbits.export_datasets().is_empty());
}
