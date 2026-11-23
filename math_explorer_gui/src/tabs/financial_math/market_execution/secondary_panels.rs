//! Spread/total middle and live/futures hedge sub-panels for [`super::MarketExecutionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MarketExecutionTool;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, Plot, PlotPoint};
use math_explorer::pure_math::statistics::kelly::{Odds, UnitInterval};
use math_explorer::pure_math::statistics::market_pricing::{
    evaluate_hedge, evaluate_middle, HedgeMode, HedgeReport, MiddleReport,
};

pub(super) fn render_middle_panel(
    tool: &mut MarketExecutionTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    render_middle_controls(tool, ui);
    let (Ok(oa), Ok(ob), Ok(p_mid)) = (
        Odds::new(tool.middle_odds_a),
        Odds::new(tool.middle_odds_b),
        UnitInterval::new((tool.middle_hit_prob_pct / 100.0).clamp(0.0, 1.0)),
    ) else {
        ui.colored_label(
            palette.error,
            "Enter valid Side A / Side B odds and middle probability.",
        );
        return;
    };
    if let Ok(rep) = evaluate_middle(oa, ob, tool.budget, Some(p_mid)) {
        render_middle_summary_and_plot(&rep, ui, palette);
    }
}

fn render_middle_controls(tool: &mut MarketExecutionTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Total Budget ($):");
        ui.add(
            egui::DragValue::new(&mut tool.budget)
                .speed(10.0)
                .range(1.0..=1_000_000.0)
                .prefix("$"),
        );
        ui.label("Side A Decimal Odds (e.g. -2.5):");
        ui.add(
            egui::DragValue::new(&mut tool.middle_odds_a)
                .speed(0.01)
                .range(1.01..=1000.0)
                .fixed_decimals(4),
        );
        ui.label("Side B Decimal Odds (e.g. +3.5):");
        ui.add(
            egui::DragValue::new(&mut tool.middle_odds_b)
                .speed(0.01)
                .range(1.01..=1000.0)
                .fixed_decimals(4),
        );
        ui.label("Est. Middle Hit %:");
        ui.add(
            egui::DragValue::new(&mut tool.middle_hit_prob_pct)
                .speed(0.25)
                .range(0.0..=100.0)
                .suffix("%"),
        );
    });
}

fn render_middle_summary_and_plot(
    rep: &MiddleReport,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let ev = rep.expected_profit.unwrap_or(0.0);
    let col = if ev >= 0.0 {
        palette.success
    } else {
        palette.warning
    };
    ui.colored_label(
        col,
        format!(
            "Stake A = ${:.2} | Stake B = ${:.2} | Non-Middle Loss = -${:.2} | Middle Jackpot = +${:.2} | Break-Even Hit Rate = {:.2}% | Expected P&L = {:+.2}",
            rep.stake_a,
            rep.stake_b,
            rep.qualifying_loss,
            rep.middle_profit,
            rep.break_even_probability * 100.0,
            ev
        ),
    );
    let pts: Vec<PlotPoint> = (0..=50)
        .map(|k| {
            let p = (k as f64) * 0.005;
            let pnl = (1.0 - p) * rep.side_a_only_pnl + p * rep.middle_profit;
            PlotPoint::new(p * 100.0, pnl)
        })
        .collect();
    Plot::new("middle_ev_curve")
        .height(210.0)
        .legend(Legend::default())
        .x_axis_label("Middle Hit Probability (%)")
        .y_axis_label("Expected Profit ($)")
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new("Expected Profit ($)", &pts[..]).color(palette.series[0]));
            plot_ui.hline(HLine::new("Break-Even ($0)", 0.0).color(palette.graph_edge));
        });
}

pub(super) fn render_hedge_panel(
    tool: &mut MarketExecutionTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    render_hedge_controls(tool, ui);
    let (Ok(o0), Ok(oh)) = (
        Odds::new(tool.hedge_orig_odds),
        Odds::new(tool.hedge_live_odds),
    ) else {
        ui.colored_label(palette.error, "Invalid original or live hedge odds.");
        return;
    };
    let mode = if tool.hedge_use_target_floor {
        HedgeMode::TargetFloor(tool.hedge_target_floor)
    } else {
        HedgeMode::EqualProfit
    };
    match evaluate_hedge(tool.hedge_orig_stake, o0, oh, mode) {
        Ok(rep) => {
            ui.colored_label(
                palette.success,
                format!(
                    "Recommended Hedge Stake = ${:.2} | Original Wins = {:+.2} | Hedge Wins = {:+.2} | Guaranteed Floor = {:+.2} (Equal-Profit Ceiling: {:+.2})",
                    rep.hedge_stake,
                    rep.net_profit_if_original_wins,
                    rep.net_profit_if_hedge_wins,
                    rep.guaranteed_floor,
                    rep.equal_profit_ceiling
                ),
            );
            render_hedge_payoff_chart(&rep, ui, palette);
        }
        Err(e) => {
            ui.colored_label(palette.error, format!("{e}"));
        }
    }
}

fn render_hedge_controls(tool: &mut MarketExecutionTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Original Stake ($):");
        ui.add(
            egui::DragValue::new(&mut tool.hedge_orig_stake)
                .speed(5.0)
                .range(1.0..=1_000_000.0)
                .prefix("$"),
        );
        ui.label("Original Decimal Odds:");
        ui.add(
            egui::DragValue::new(&mut tool.hedge_orig_odds)
                .speed(0.1)
                .range(1.01..=1000.0)
                .fixed_decimals(3),
        );
        ui.label("Live Hedge Odds:");
        ui.add(
            egui::DragValue::new(&mut tool.hedge_live_odds)
                .speed(0.02)
                .range(1.01..=1000.0)
                .fixed_decimals(4),
        );
        ui.checkbox(&mut tool.hedge_use_target_floor, "Target Floor ($)");
        if tool.hedge_use_target_floor {
            ui.add(
                egui::DragValue::new(&mut tool.hedge_target_floor)
                    .speed(5.0)
                    .range(-100_000.0..=100_000.0)
                    .prefix("$"),
            );
        }
    });
}

fn render_hedge_payoff_chart(rep: &HedgeReport, ui: &mut egui::Ui, palette: &SemanticPalette) {
    let h_max = ((rep.original_stake * rep.original_odds.value()) / rep.hedge_odds.value() * 1.35)
        .max(10.0);
    let mut orig_win_pts = Vec::with_capacity(41);
    let mut hedge_win_pts = Vec::with_capacity(41);
    for k in 0..=40 {
        let h = h_max * (k as f64) / 40.0;
        orig_win_pts.push(PlotPoint::new(
            h,
            rep.original_stake * (rep.original_odds.value() - 1.0) - h,
        ));
        hedge_win_pts.push(PlotPoint::new(
            h,
            h * (rep.hedge_odds.value() - 1.0) - rep.original_stake,
        ));
    }
    Plot::new("hedge_payoff_lines")
        .height(210.0)
        .legend(Legend::default())
        .x_axis_label("Hedge Stake H ($)")
        .y_axis_label("Net Profit ($)")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("If Original Wins ($)", &orig_win_pts[..]).color(palette.series[0]),
            );
            plot_ui.line(
                Line::new("If Hedge Wins ($)", &hedge_win_pts[..]).color(palette.series[1]),
            );
        });
}

// [cite:clinical_trials_statistics]
