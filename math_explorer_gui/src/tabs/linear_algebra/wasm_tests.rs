#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::PRESET_LINALG_LAUCHLI;
use super::state::ConditioningWorkbenchTool;
use super::LinearAlgebraTab;
use crate::framework::InteractiveTool;
use crate::tabs::ExplorerTab;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_linear_algebra_tab_and_conditioning_workbench() {
    let tab = LinearAlgebraTab::default();
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "linear_algebra");

    let mut tool = ConditioningWorkbenchTool::default();
    tool.import_preset_json(PRESET_LINALG_LAUCHLI)
        .expect("valid Lauchli preset");
    assert!(tool.snapshot.solvers[1].forward_rel_error < 1e-6);
    assert!(tool.snapshot.solvers[2].forward_rel_error < 1e-6);
}

// [cite:advanced_linear_algebra]
