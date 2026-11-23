#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::bayesian_tool::BayesianInferenceTool;
use super::copula_tool::{CopulaDependenceTool, CopulaToolConfig};
use super::distribution_tool::{DistributionExplorerTool, DistributionFamily};
use super::presets::{
    PRESET_BAYES_CLINICAL_RESPONDER, PRESET_COPULA_4D_QMC, PRESET_DIST_GAMMA_SURVIVAL,
};
use super::ProbabilityStatisticsTab;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::tabs::ExplorerTab;
use eframe::egui;

#[test]
fn test_probability_statistics_tab_slug_domain_and_all_three_tools() {
    let mut tab = ProbabilityStatisticsTab::default();
    assert_eq!(tab.name(), "Probability & Statistics");
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "probability_statistics");

    for (id, display) in [
        ("DistributionExplorerTool", "Distribution Explorer"),
        ("BayesianInferenceTool", "Bayesian Inference Lab"),
        ("CopulaDependenceTool", "Multivariate Copula Lab"),
    ] {
        assert!(tab.select_tool_by_name(id));
        assert_eq!(tab.active_tool_id(), Some(id));
        assert!(tab.select_tool_by_name(display));
        assert_eq!(tab.active_tool_id(), Some(id));
    }
}

#[test]
fn test_all_three_workbenches_headless_egui_accesskit_and_png_export() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1360.0, 840.0),
        )),
        ..Default::default()
    };

    let mut dist = DistributionExplorerTool::default();
    dist.import_preset_json(PRESET_DIST_GAMMA_SURVIVAL).unwrap();
    for fam in DistributionFamily::CONTINUOUS
        .into_iter()
        .chain(DistributionFamily::DISCRETE)
    {
        let mut cfg = dist.config.clone();
        cfg.family = fam;
        let (a, b) = fam.default_params();
        cfg.param_a = a;
        cfg.param_b = b;
        dist.apply_config(cfg).unwrap();
        assert!(!dist.snapshot.cdf_curve.is_empty());
    }
    let out_dist = ctx.run(raw.clone(), |c| dist.show(c));
    assert!(out_dist.platform_output.accesskit_update.is_some());

    let mut bayes = BayesianInferenceTool::default();
    bayes
        .import_preset_json(PRESET_BAYES_CLINICAL_RESPONDER)
        .unwrap();
    bayes.step_mcmc_draws(0, 64);
    assert!((bayes.mcmc_summary.mean - bayes.analytical.posterior_mean).abs() < 0.08);
    let out_bayes = ctx.run(raw.clone(), |c| bayes.show(c));
    assert!(out_bayes.platform_output.accesskit_update.is_some());

    let mut copula = CopulaDependenceTool::default();
    copula.import_preset_json(PRESET_COPULA_4D_QMC).unwrap();
    assert!(copula.snapshot.cholesky_error.is_none());
    let joint = copula.snapshot.active_joint_cdf.expect("4D QMC joint CDF");
    assert!(joint >= copula.snapshot.frechet_bounds.0 - 1e-6);
    assert!(joint <= copula.snapshot.frechet_bounds.1 + 1e-6);
    let out_cop = ctx.run(raw, |c| copula.show(c));
    assert!(out_cop.platform_output.accesskit_update.is_some());

    dist.import_preset_json(PRESET_DIST_GAMMA_SURVIVAL).unwrap();
    let opts = egui_plot::export::ExportOptions {
        width: 1200,
        height: 720,
        ..Default::default()
    };
    for (path, datasets) in [
        ("/tmp/prob_stat_distributions.png", dist.export_datasets()),
        ("/tmp/prob_stat_bayesian_mcmc.png", bayes.export_datasets()),
        ("/tmp/prob_stat_copula_qmc.png", copula.export_datasets()),
    ] {
        let png = egui_plot::export::export_custom_png(&datasets, &opts).unwrap();
        assert!(!png.is_empty());
        let _ = std::fs::write(path, &png);
    }
}

#[test]
fn test_copula_non_spd_equicorrelation_and_finite_state_rollback() {
    let mut copula = CopulaDependenceTool::default();
    // For d = 4, equicorrelation is SPD iff rho > -1/3 = -0.3333...
    copula
        .apply_config(CopulaToolConfig {
            correlation_rho: -0.42,
            marginal_u1: 0.60,
            marginal_u2: 0.70,
            zip_zero_prob: 0.35,
            dimension: 4,
            qmc_samples: 128,
            seed: 12,
        })
        .unwrap();
    assert!(copula.snapshot.cholesky_error.is_some());
    assert!(copula.snapshot.active_joint_cdf.is_none());

    let mut fw = SimulationFramework::new("probability_statistics");
    assert!(fw.select_tool_by_name("BayesianInferenceTool"));
    let baseline = fw.active_tool.as_ref().unwrap().save_state().unwrap();
    fw.cache_valid_snapshot();
    fw.active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<BayesianInferenceTool>()
        .unwrap()
        .config
        .proposal_scale = f64::NAN;
    assert!(fw.inspect_runtime_state().is_err());
    assert!(fw.rollback_step_back());
    assert_eq!(fw.active_tool.as_ref().unwrap().save_state().unwrap(), baseline);
}

// [cite:clinical_trials_statistics]
