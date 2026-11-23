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

    let fragment = tab.encode_url_share_fragment().unwrap();
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
