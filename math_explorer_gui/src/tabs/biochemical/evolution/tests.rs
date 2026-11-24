#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{PRESET_HAWK_DOVE_ESS, PRESET_HAWK_ESCALATION};
use super::EvolutionaryBiologyTool;
use crate::framework::InteractiveTool;
use crate::tabs::biochemical::BiochemicalTab;
use crate::tabs::ExplorerTab;

#[test]
fn test_evolutionary_biology_converges_to_ess_and_matches_fermi_preset() {
    let mut tool = EvolutionaryBiologyTool::default();
    let ess = tool.population().ess_hawk_frequency();
    assert!((ess - 0.5).abs() < 1e-12);
    let final_p = tool.primary_hawk_traj.last().unwrap()[1];
    assert!(
        (final_p - ess).abs() < 1e-2,
        "Expected convergence to p* = 0.5, got {final_p}"
    );

    #[cfg(feature = "physics")]
    {
        use crate::tabs::solid_state::ising::FermiGamePreset;
        assert_eq!(
            FermiGamePreset::HawkDove.payoffs(),
            tool.population().cooperator_defector_payoffs()
        );
        assert_eq!(FermiGamePreset::HawkDove.payoffs(), [1.0, 0.0, 2.0, -1.0]);
    }

    tool.import_preset_json(PRESET_HAWK_DOVE_ESS).unwrap();
    let final_third = tool.primary_hawk_traj.last().unwrap()[1];
    assert!((final_third - (1.0 / 3.0)).abs() < 1e-2);

    tool.import_preset_json(PRESET_HAWK_ESCALATION).unwrap();
    let final_pure = tool.primary_hawk_traj.last().unwrap()[1];
    assert!(final_pure > 0.99);
}

#[test]
fn test_biochemical_tab_url_slug_and_evolution_share_fragment_roundtrip() {
    let mut tab = BiochemicalTab::default();
    assert_eq!(tab.url_slug(), "biochemical");
    assert_eq!(tab.academic_domain(), "Biology");
    assert!(tab.select_tool_by_name("EvolutionaryBiologyTool"));

    let preset_json = tab.export_url_preset_json().unwrap();
    let fragment = crate::url_state::encode_share_fragment(
        tab.url_slug(),
        tab.active_tool_id(),
        preset_json.as_deref(),
    )
    .unwrap();
    assert!(
        fragment.starts_with("#tab=biochemical&tool=EvolutionaryBiologyTool"),
        "Unexpected fragment: {fragment}"
    );

    let parsed = crate::url_state::parse_share_location(&fragment, "")
        .unwrap()
        .expect("share location must parse");
    let mut restored_tab = BiochemicalTab::default();
    restored_tab.import_url_share_params(&parsed, None).unwrap();
    assert_eq!(
        restored_tab.active_tool_id(),
        Some("EvolutionaryBiologyTool")
    );
}

#[test]
fn test_biology_studios_portal_paper_links_resolve_computational_biology() {
    use crate::framework::theory_links::resolve_portal_paper_links;

    for (tool_id, domain) in [
        ("PhyllotaxisLSystemTool", "biology"),
        ("BiochemicalDynamicsTool", "biochemical"),
        ("EvolutionaryBiologyTool", "biochemical"),
    ] {
        let links = resolve_portal_paper_links(tool_id, domain);
        assert!(
            links
                .iter()
                .any(|p| p.paper_name == "computational_biology.tex"),
            "Expected computational_biology.tex link for {tool_id} ({domain}), got {links:?}"
        );
    }
}

#[test]
fn test_biochemical_multi_pane_workspace_url_roundtrip() {
    use crate::app::{MathExplorerApp, ViewportLayout};

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let _ = crate::url_state::take_last_copied_share_url();
    let ctx = eframe::egui::Context::default();
    let mut app = MathExplorerApp::default();
    let idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "biochemical")
        .unwrap();

    app.selected_tab = idx;
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(1, idx);
    app.focus_pane(1);
    assert!(app
        .pane_tab_mut(0)
        .unwrap()
        .select_tool_by_name("BiochemicalDynamicsTool"));
    assert!(app
        .pane_tab_mut(1)
        .unwrap()
        .select_tool_by_name("EvolutionaryBiologyTool"));
    app.pane_tab_mut(1)
        .unwrap()
        .framework_mut()
        .unwrap()
        .import_url_preset_json(Some("EvolutionaryBiologyTool"), Some(PRESET_HAWK_DOVE_ESS))
        .unwrap();

    app.copy_current_share_link(&ctx);
    let copied = crate::url_state::take_last_copied_share_url().unwrap();
    let ws_hash = &copied[copied.find('#').unwrap()..];
    assert!(
        ws_hash.starts_with("#ws=") && ws_hash.len() <= crate::url_state::MAX_URL_FRAGMENT_BYTES
    );

    let mut restored = MathExplorerApp::default();
    assert!(restored
        .apply_share_location_strings(ws_hash, "", Some(&ctx))
        .unwrap());
    assert_eq!(restored.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(restored.focused_pane, 1);
    assert_eq!(
        restored.pane_tab(0).unwrap().active_tool_id(),
        Some("BiochemicalDynamicsTool")
    );
    assert_eq!(
        restored.pane_tab(1).unwrap().active_tool_id(),
        Some("EvolutionaryBiologyTool")
    );
    let restored_json = restored
        .pane_tab(1)
        .unwrap()
        .export_url_preset_json()
        .unwrap()
        .unwrap();
    assert!(
        restored_json.contains("6.0"),
        "Expected injury_cost 6.0 in {restored_json}"
    );
}
