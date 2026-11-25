//! Unit and rendering tests for `FinancialMathTab` market pricing calculators.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::market_devig::MarketDeviggingTool;
use super::market_execution::{ExecutionCalculatorMode, MarketExecutionTool};
use super::FinancialMathTab;
use crate::framework::{ExportFormat, InteractiveTool, SimulationFramework};
use crate::tabs::ExplorerTab;
use eframe::egui;

fn smoke_check_tool<T: InteractiveTool + Default>(ctx: &egui::Context) {
    let mut tool = T::default();
    let _ = tool.name();
    let _ = tool.theory();
    let _ = tool.export_datasets();
    let _ = tool.export_figure_options();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
}

pub(crate) fn check_financial_math_tools(ctx: &egui::Context) {
    smoke_check_tool::<super::bet_size::BetSizeCalculatorTool>(ctx);
    smoke_check_tool::<super::bankroll_growth::BankrollGrowthTool>(ctx);
    smoke_check_tool::<super::correlated_risk::CorrelatedRiskScenarioTool>(ctx);
    smoke_check_tool::<MarketDeviggingTool>(ctx);
    smoke_check_tool::<MarketExecutionTool>(ctx);
    smoke_check_tool::<super::monte_carlo_workbench::MonteCarloWorkbenchTool>(ctx);
    smoke_check_tool::<super::portfolio_workbench::PortfolioCopulaWorkbenchTool>(ctx);
    smoke_check_tool::<super::tail_risk::TailRiskDrawdownTool>(ctx);
    #[cfg(feature = "applied")]
    smoke_check_tool::<super::markowitz::MarkowitzFrontierTool>(ctx);
}

fn run_export_frame(framework: &mut SimulationFramework, format: ExportFormat) {
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 860.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        egui_plot::clear_last_plot_datasets();
        framework.show_export_modal = true;
        framework.queue_export(format);
        framework.show(ctx, "financial_math_test");
    });
}

#[test]
fn test_financial_math_tab_selects_market_pricing_tools() {
    let mut tab = FinancialMathTab::default();
    assert!(tab.select_tool_by_name("Fair Odds & +EV Devigger"));
    assert!(tab.select_tool_by_name("Arbitrage, Dutching & Hedge"));
    assert!(tab.select_tool_by_name("Tail Risk & Drawdown"));
    assert!(tab.select_tool_by_name("Portfolio & Copula Risk Workbench"));
    #[cfg(feature = "applied")]
    assert!(tab.select_tool_by_name("Markowitz Efficient Frontier"));
    let saved = tab.save_state().expect("tab state serializes");
    tab.load_state(&saved);
}

#[cfg(feature = "applied")]
#[test]
fn test_financial_math_workbenches_deferred_csv_json_png_svg_exports() {
    for id in [
        "BetSizeCalculatorTool",
        "BankrollGrowthTool",
        "CorrelatedRiskScenarioTool",
        "PortfolioCopulaWorkbenchTool",
        "TailRiskDrawdownTool",
        "MarkowitzFrontierTool",
    ] {
        let meta = inventory::iter::<crate::framework::ToolMetadata>
            .into_iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("Missing financial_math tool {id}"));
        let mut fw = SimulationFramework::new("__fin_export_audit__");
        fw.active_tool = Some((meta.build)());

        run_export_frame(&mut fw, ExportFormat::Csv);
        let csv = fw.export_queue.last_exported_content.as_deref();
        assert!(
            csv.is_some_and(|s| s.lines().count() >= 5),
            "Short CSV for {id}"
        );

        run_export_frame(&mut fw, ExportFormat::Json);
        let json = fw.export_queue.last_exported_content.as_deref();
        assert!(json.is_some_and(|s| serde_json::from_str::<serde_json::Value>(s).is_ok()));

        run_export_frame(&mut fw, ExportFormat::Png);
        assert!(fw
            .export_queue
            .last_exported_bytes
            .as_deref()
            .is_some_and(|b| b.len() > 200));

        run_export_frame(&mut fw, ExportFormat::Svg);
        assert!(fw
            .export_queue
            .last_exported_content
            .as_deref()
            .is_some_and(|s| s.contains("<svg")));
    }
}

#[test]
fn test_market_devigger_renders_1x2_and_6way_simplex_guard_with_exports() {
    let mut framework = SimulationFramework::new("__devig_test__");
    framework.active_tool = Some(Box::new(MarketDeviggingTool::default()));

    run_export_frame(&mut framework, ExportFormat::Csv);
    let csv = framework
        .export_queue
        .last_exported_content
        .clone()
        .expect("CSV exported");
    assert!(csv.contains("Raw Implied %"));
    assert!(csv.contains("Shin %"));

    run_export_frame(&mut framework, ExportFormat::Png);
    let png = framework
        .export_queue
        .last_exported_bytes
        .clone()
        .expect("PNG exported");
    assert!(png.len() > 200);
    let _ = std::fs::write("/tmp/market_devig_probabilities.png", &png);

    let mut six_way = MarketDeviggingTool::default();
    six_way
        .try_load_state(
            r#"{"bankroll":1500.0,"quote_format":"Decimal","method":"Additive","decimal_quotes":[1.9,3.25,6.5,10.0,21.0,101.0],"target_outcome":5,"offered_decimal_odds":110.0}"#,
        )
        .expect("6-way state loads");
    framework.active_tool = Some(Box::new(six_way));
    run_export_frame(&mut framework, ExportFormat::Svg);
    let svg = framework
        .export_queue
        .last_exported_content
        .clone()
        .expect("SVG exported");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_market_execution_renders_all_four_modes_and_exports_figures() {
    for (mode, out_png) in [
        (
            ExecutionCalculatorMode::Arbitrage,
            Some("/tmp/market_execution_arbitrage.png"),
        ),
        (ExecutionCalculatorMode::Dutching, None),
        (ExecutionCalculatorMode::Middle, None),
        (
            ExecutionCalculatorMode::Hedge,
            Some("/tmp/market_execution_hedge_payoff.png"),
        ),
    ] {
        let mut tool = MarketExecutionTool::default();
        let mut json_val: serde_json::Value =
            serde_json::from_str(&tool.save_state().expect("state")).expect("valid json");
        json_val["mode"] = serde_json::to_value(mode).expect("mode json");
        tool.try_load_state(&json_val.to_string())
            .expect("mode state loads");

        let mut framework = SimulationFramework::new("__exec_test__");
        framework.active_tool = Some(Box::new(tool));
        run_export_frame(&mut framework, ExportFormat::Png);
        let png = framework
            .export_queue
            .last_exported_bytes
            .clone()
            .expect("execution PNG exported");
        assert!(png.len() > 200);
        if let Some(path) = out_png {
            let _ = std::fs::write(path, &png);
        }
    }
}
