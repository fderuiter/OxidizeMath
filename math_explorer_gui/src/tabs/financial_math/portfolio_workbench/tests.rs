#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::io::{MAX_IMPORT_BYTES, MAX_IMPORT_ROWS};
use super::state::{AssetConfig, WorkbenchView};
use super::PortfolioCopulaWorkbenchTool;
use crate::framework::InteractiveTool;

#[test]
fn test_default_three_asset_rho_060_benchmark_metrics() {
    let tool = PortfolioCopulaWorkbenchTool::default();
    let cov_sol = tool
        .kelly_cache
        .cov_kelly
        .as_ref()
        .expect("covariance kelly computed");
    let ind_sol = tool
        .kelly_cache
        .indep_kelly
        .as_ref()
        .expect("independent kelly computed");
    let tail = tool
        .kelly_cache
        .cov_sim_tail
        .expect("simulated tail risk computed");

    // 3-asset rho=0.60 benchmark: 2.7273x gross leverage, +10.91% excess growth, ~$550 95% CVaR
    assert!((cov_sol.metrics.gross_leverage - 30.0 / 11.0).abs() < 1e-6);
    assert!((cov_sol.metrics.excess_log_growth - 1.2 / 11.0).abs() < 1e-6);
    assert!((ind_sol.metrics.gross_leverage - 6.0).abs() < 1e-6);
    assert!((ind_sol.metrics.excess_log_growth - (-0.048)).abs() < 1e-6);
    assert!(
        (tail.cvar_95 - 550.08).abs() < 25.0,
        "got cvar_95 = {}",
        tail.cvar_95
    );

    // 3-leg copula SGP (u=0.55 -> ~0.3272) and 5% crash (u=0.05 -> ~0.00767, ~61.35x naive)
    assert!((tool.copula_cache.active_sgp_prob - 0.3272).abs() < 0.006);
    assert!((tool.copula_cache.active_crash_prob - 0.00767).abs() < 0.0015);
    assert!((tool.copula_cache.crash_multiplier - 61.35).abs() < 5.0);
}

#[test]
fn test_all_four_views_export_non_empty_datasets_and_render_cleanly() {
    let mut tool = PortfolioCopulaWorkbenchTool::default();
    let ctx = eframe::egui::Context::default();
    let screen_rect = Some(eframe::egui::Rect::from_min_size(
        eframe::egui::Pos2::ZERO,
        eframe::egui::vec2(1280.0, 800.0),
    ));

    for view in [
        WorkbenchView::KellyFrontier,
        WorkbenchView::CopulaTailCrash,
        WorkbenchView::OuMeanReversion,
        WorkbenchView::BlackScholesGreeks,
    ] {
        tool.state.view = view;
        let datasets = tool.export_datasets();
        assert!(!datasets.is_empty(), "empty datasets for {view:?}");
        assert!(datasets
            .iter()
            .all(|(_, pts)| !pts.is_empty()
                && pts.iter().all(|[x, y]| x.is_finite() && y.is_finite())));

        let _ = ctx.run(
            eframe::egui::RawInput {
                screen_rect,
                ..Default::default()
            },
            |ctx| tool.show(ctx),
        );
    }
}

#[test]
fn test_csv_and_json_return_import_and_bounds_enforcement() {
    let mut tool = PortfolioCopulaWorkbenchTool::default();
    let csv = "\u{feff}Date,Tech,Rates,Gold\n2026-01-01,0.012,-0.004,0.003\n2026-01-02,-0.008,0.005,0.002\n2026-01-03,0.015,0.001,-0.003\n2026-01-04,0.004,-0.002,0.006\n2026-01-05,-0.011,0.003,0.004\n";
    tool.import_return_series_str("sample.csv", csv)
        .expect("valid CSV import");
    assert_eq!(tool.state.assets.len(), 3);
    assert_eq!(tool.state.assets[0].label, "Tech");
    assert!(tool.kelly_cache.empirical_tail.is_some());

    let json = r#"{"columns":["Alpha","Beta"],"returns":[[0.01,0.02],[-0.01,0.005],[0.015,-0.004],[0.008,0.011]]}"#;
    tool.import_return_series_str("sample.json", json)
        .expect("valid JSON import");
    assert_eq!(tool.state.assets.len(), 2);

    // Enforce > 2 MiB byte cap
    let oversized = "0.01,0.02\n".repeat((MAX_IMPORT_BYTES / 10) + 10);
    assert!(tool
        .import_return_series_str("huge.csv", &oversized)
        .is_err());

    // Enforce > 10,000 row cap
    let too_many_rows = "0.1,0.2\n".repeat(MAX_IMPORT_ROWS + 2);
    assert!(tool
        .import_return_series_str("rows.csv", &too_many_rows)
        .is_err());
}

#[test]
fn test_tearsheet_exports_and_12_asset_url_state_budget() {
    let mut tool = PortfolioCopulaWorkbenchTool::default();
    tool.state.assets[0].label = "=SUM(A1:A2)".to_string();
    let csv = tool.export_tearsheet_csv();
    assert!(
        csv.contains("\"'=SUM(A1:A2)\""),
        "CSV tear-sheet must neutralize formula injection: {csv}"
    );
    assert!(csv.contains("VaR_95_USD") && csv.contains("Black_Scholes"));

    let json = tool.export_tearsheet_json();
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid tearsheet JSON");
    assert!(parsed.get("kelly_weights").is_some());
    assert!(parsed.get("tail_risk_usd").is_some());
    assert!(parsed.get("ou_analyzer").is_some());
    assert!(parsed.get("black_scholes_greeks").is_some());

    // Verify 12-asset URL state stays well below 4,096 bytes
    tool.state.assets = (0..12).map(AssetConfig::benchmark_leg).collect();
    tool.state.copula_legs = 12;
    tool.recalculate_all();
    let saved = tool.save_state().expect("serializes 12-asset state");
    assert!(
        saved.len() < 2_048,
        "12-asset state must fit comfortably inside 4,096-byte URL budget (got {} bytes)",
        saved.len()
    );
    let mut restored = PortfolioCopulaWorkbenchTool::default();
    restored
        .try_load_state(&saved)
        .expect("roundtrips 12-asset state");
    assert_eq!(restored.state.assets.len(), 12);
}
