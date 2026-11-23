//! `egui` rendering for [`super::MarketExecutionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::secondary_panels::{render_hedge_panel, render_middle_panel};
use super::{ExecutionCalculatorMode, MarketExecutionTool, MAX_EXEC_LEGS, MIN_EXEC_LEGS};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Bar, BarChart, HLine, Legend, Plot};
use math_explorer::pure_math::statistics::kelly::Odds;
use math_explorer::pure_math::statistics::market_pricing::{
    booksum, dutch_subset_into, evaluate_arbitrage_into, DutchingMode, DutchingReport,
    MAX_MARKET_OUTCOMES,
};

pub(super) fn render_execution_tool(tool: &mut MarketExecutionTool, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    egui::CentralPanel::default().show_inside(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            render_mode_selector(tool, ui);
            ui.separator();
            match tool.mode {
                ExecutionCalculatorMode::Arbitrage => render_arbitrage_panel(tool, ui, &palette),
                ExecutionCalculatorMode::Dutching => render_dutching_panel(tool, ui, &palette),
                ExecutionCalculatorMode::Middle => render_middle_panel(tool, ui, &palette),
                ExecutionCalculatorMode::Hedge => render_hedge_panel(tool, ui, &palette),
            }
        });
    });
}

fn render_mode_selector(tool: &mut MarketExecutionTool, ui: &mut egui::Ui) {
    ui.heading("Cross-Market Arbitrage, Subset Dutching, Middle & Live Hedge");
    ui.horizontal_wrapped(|ui| {
        ui.label("Calculator:");
        ui.radio_value(
            &mut tool.mode,
            ExecutionCalculatorMode::Arbitrage,
            "Cross-Book Arbitrage",
        );
        ui.radio_value(
            &mut tool.mode,
            ExecutionCalculatorMode::Dutching,
            "Subset Dutching",
        );
        ui.radio_value(
            &mut tool.mode,
            ExecutionCalculatorMode::Middle,
            "Spread / Total Middle",
        );
        ui.radio_value(
            &mut tool.mode,
            ExecutionCalculatorMode::Hedge,
            "Live / Futures Hedge",
        );
    });
}

fn parse_odds_slice(raw: &[f64]) -> Result<Vec<Odds>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for &q in raw {
        out.push(Odds::new(q).map_err(|e| format!("Invalid odds {q:.3}: {e}"))?);
    }
    Ok(out)
}

fn render_arbitrage_panel(
    tool: &mut MarketExecutionTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    render_arbitrage_controls(tool, ui);
    let Ok(odds) = parse_odds_slice(&tool.arb_odds) else {
        ui.colored_label(palette.error, "Enter valid decimal odds > 1.00 on all legs.");
        return;
    };
    let mut stakes = [0.0; MAX_MARKET_OUTCOMES];
    match evaluate_arbitrage_into(&odds, tool.budget, &mut stakes[..odds.len()]) {
        Ok(rep) => {
            ui.colored_label(
                palette.success,
                format!(
                    "✓ Risk-Free Arbitrage Detected! Reciprocal Sum = {:.4}% | Guaranteed Payout = ${:.2} | Guaranteed Profit = +${:.2} ({:+.2}% ROI)",
                    rep.reciprocal_sum * 100.0,
                    rep.guaranteed_payout,
                    rep.guaranteed_profit,
                    rep.roi * 100.0
                ),
            );
            render_arb_bars(&stakes[..odds.len()], rep.guaranteed_profit, ui, palette);
        }
        Err(_) => {
            let r = booksum(&odds).unwrap_or(1.0);
            ui.colored_label(
                palette.warning,
                format!(
                    "No Arbitrage: combined implied probability is {:.2}% (>= 100.00%, synthetic hold {:+.2}%).",
                    r * 100.0,
                    (r - 1.0) * 100.0
                ),
            );
        }
    }
}

fn render_arbitrage_controls(tool: &mut MarketExecutionTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Total Budget ($):");
        ui.add(
            egui::DragValue::new(&mut tool.budget)
                .speed(25.0)
                .range(1.0..=1_000_000.0)
                .prefix("$"),
        );
        if ui.button("− Leg").clicked() && tool.arb_odds.len() > MIN_EXEC_LEGS {
            tool.arb_odds.pop();
        }
        if ui.button("+ Leg").clicked() && tool.arb_odds.len() < MAX_EXEC_LEGS {
            tool.arb_odds.push(3.20);
        }
        if ui.button("Preset: 2.10 / 1.98 ($1,000)").clicked() {
            tool.budget = 1000.0;
            tool.arb_odds = vec![2.10, 1.980_392];
        }
    });
    ui.horizontal_wrapped(|ui| {
        for (i, q) in tool.arb_odds.iter_mut().enumerate() {
            ui.group(|ui| {
                ui.label(format!("Leg #{} Best Odds", i + 1));
                ui.add(
                    egui::DragValue::new(q)
                        .speed(0.01)
                        .range(1.01..=1000.0)
                        .fixed_decimals(4),
                );
            });
        }
    });
}

fn render_arb_bars(stakes: &[f64], profit: f64, ui: &mut egui::Ui, palette: &SemanticPalette) {
    ui.horizontal_wrapped(|ui| {
        for (i, &s) in stakes.iter().enumerate() {
            ui.label(format!("Leg #{} Stake: ${:.2}", i + 1, s));
            ui.separator();
        }
    });
    let bars: Vec<Bar> = stakes
        .iter()
        .enumerate()
        .map(|(i, &s)| Bar::new((i + 1) as f64, s).width(0.45))
        .collect();
    Plot::new("arb_stakes_chart")
        .height(210.0)
        .legend(Legend::default())
        .x_axis_label("Outcome Leg")
        .y_axis_label("Dollars ($)")
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new("Allocated Stake ($)", bars).color(palette.series[0]));
            plot_ui.hline(
                HLine::new("Guaranteed Net Profit ($)", profit).color(palette.success),
            );
        });
}

fn render_dutching_panel(
    tool: &mut MarketExecutionTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(
            &mut tool.dutch_target_profit_mode,
            "Size for Target Profit (instead of Fixed Budget)",
        );
        if tool.dutch_target_profit_mode {
            ui.label("Target Profit ($):");
            ui.add(
                egui::DragValue::new(&mut tool.dutch_target_profit)
                    .speed(5.0)
                    .range(1.0..=1_000_000.0)
                    .prefix("$"),
            );
        } else {
            ui.label("Fixed Budget ($):");
            ui.add(
                egui::DragValue::new(&mut tool.budget)
                    .speed(5.0)
                    .range(1.0..=1_000_000.0)
                    .prefix("$"),
            );
        }
        if ui.button("− Contender").clicked() && tool.dutch_odds.len() > MIN_EXEC_LEGS {
            tool.dutch_odds.pop();
            tool.dutch_selected.pop();
        }
        if ui.button("+ Contender").clicked() && tool.dutch_odds.len() < MAX_EXEC_LEGS {
            tool.dutch_odds.push(8.0);
            tool.dutch_selected.push(true);
        }
    });
    ui.horizontal_wrapped(|ui| {
        for i in 0..tool.dutch_odds.len() {
            ui.group(|ui| {
                ui.checkbox(&mut tool.dutch_selected[i], format!("Contender #{}", i + 1));
                ui.add(
                    egui::DragValue::new(&mut tool.dutch_odds[i])
                        .speed(0.05)
                        .range(1.01..=1000.0)
                        .fixed_decimals(2),
                );
            });
        }
    });
    render_dutching_output(tool, ui, palette);
}

fn render_dutching_output(
    tool: &MarketExecutionTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let Ok(odds) = parse_odds_slice(&tool.dutch_odds) else {
        ui.colored_label(palette.error, "Invalid contender odds.");
        return;
    };
    let indices: Vec<usize> = tool
        .dutch_selected
        .iter()
        .enumerate()
        .filter_map(|(i, &sel)| if sel { Some(i) } else { None })
        .collect();
    let mode = if tool.dutch_target_profit_mode {
        DutchingMode::TargetProfit(tool.dutch_target_profit)
    } else {
        DutchingMode::FixedBudget(tool.budget)
    };
    let mut stakes = [0.0; MAX_MARKET_OUTCOMES];
    match dutch_subset_into(&odds, &indices, mode, &mut stakes[..odds.len()]) {
        Ok(rep) => {
            render_dutching_summary_and_chart(tool, &stakes[..odds.len()], &rep, ui, palette);
        }
        Err(e) => {
            ui.colored_label(palette.error, format!("Dutching infeasible: {e}"));
        }
    }
}

fn render_dutching_summary_and_chart(
    tool: &MarketExecutionTool,
    stakes: &[f64],
    rep: &DutchingReport,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let total_count = stakes.len();
    let col = if rep.is_profitable_on_hit {
        palette.success
    } else {
        palette.warning
    };
    ui.colored_label(
        col,
        format!(
            "Selected {}/{} Contenders | Combined Implied = {:.2}% | Total Stake = ${:.2} | Payout = ${:.2} | Net Profit If Hit = {:+.2} ({:+.2}%)",
            rep.selected_count,
            total_count,
            rep.combined_implied_probability * 100.0,
            rep.total_stake,
            rep.winning_payout,
            rep.profit_if_hit,
            rep.roi_if_hit * 100.0
        ),
    );
    ui.horizontal_wrapped(|ui| {
        for (i, &s) in stakes.iter().enumerate() {
            if tool.dutch_selected[i] {
                ui.label(format!("Contender #{} Stake: ${:.2}", i + 1, s));
                ui.separator();
            }
        }
    });
    let pnl_bars: Vec<Bar> = (0..total_count)
        .map(|i| {
            let net = if tool.dutch_selected[i] {
                rep.profit_if_hit
            } else {
                -rep.total_stake
            };
            Bar::new((i + 1) as f64, net).width(0.45)
        })
        .collect();
    Plot::new("dutching_pnl_chart")
        .height(210.0)
        .legend(Legend::default())
        .x_axis_label("Winning Contender")
        .y_axis_label("Net P&L ($)")
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(
                BarChart::new("Net P&L by Winning Contender ($)", pnl_bars)
                    .color(palette.series[2]),
            );
        });
}

// [cite:clinical_trials_statistics]
