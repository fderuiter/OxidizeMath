#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::bayesian_tool::BayesianInferenceTool;
use super::copula_tool::CopulaDependenceTool;
use super::distribution_tool::DistributionExplorerTool;
use super::ProbabilityStatisticsTab;
use crate::tabs::ExplorerTab;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_probability_statistics_tab_and_zero_alloc_workbenches() {
    let mut tab = ProbabilityStatisticsTab::default();
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "probability_statistics");
    assert!(tab.select_tool_by_name("BayesianInferenceTool"));

    let dist = DistributionExplorerTool::default();
    assert!(dist.snapshot.mean.is_some());

    let mut bayes = BayesianInferenceTool::default();
    bayes.step_mcmc_draws(0, 32);
    assert!(bayes.chain_len >= 32);

    let copula = CopulaDependenceTool::default();
    assert!(copula.snapshot.cholesky_error.is_none());
}

// [cite:clinical_trials_statistics]
