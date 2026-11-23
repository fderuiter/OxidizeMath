//! `egui` rendering for [`super::MarketDeviggingTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MarketDeviggingTool, QuoteFormat, MAX_GUI_OUTCOMES, MIN_OUTCOMES};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Plot};
use math_explorer::pure_math::statistics::kelly::Odds;
use math_explorer::pure_math::statistics::market_pricing::{
    compare_outcome_ev_all_methods, devig_additive_detailed_into, devig_into,
    devig_multiplicative_into, devig_power_into, devig_shin_into, AllMethodsEvReport,
    DevigMetadata, DevigMethod, MAX_MARKET_OUTCOMES,
};

fn decimal_to_american(dec: f64) -> f64 {
    if dec >= 2.0 {
        (dec - 1.0) * 100.0
    } else {
        -100.0 / (dec - 1.0).max(0.01)
    }
}

fn american_to_decimal(am: f64) -> f64 {
    if am >= 100.0 {
        1.0 + am / 100.0
    } else if am <= -100.0 {
        1.0 + 100.0 / am.abs()
    } else if am >= 0.0 {
        2.0
    } else {
        1.909_090_9
    }
}

fn edit_odds_value(ui: &mut egui::Ui, dec: &mut f64, fmt: QuoteFormat) {
    match fmt {
        QuoteFormat::Decimal => {
            ui.add(
                egui::DragValue::new(dec)
                    .speed(0.02)
                    .range(1.01..=1000.0)
                    .fixed_decimals(2),
            );
        }
        QuoteFormat::American => {
            let mut am = decimal_to_american(*dec);
            let resp = ui.add(
                egui::DragValue::new(&mut am)
                    .speed(5.0)
                    .range(-10_000.0..=10_000.0)
                    .fixed_decimals(0),
            );
            if resp.changed() && !(-99.9..100.0).contains(&am) {
                *dec = american_to_decimal(am).clamp(1.01, 1000.0);
            }
        }
    }
}

pub(super) fn render_devig_tool(tool: &mut MarketDeviggingTool, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    egui::CentralPanel::default().show_inside(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            render_header_controls(tool, ui);
            ui.separator();
            render_quote_grid(tool, ui);
            ui.separator();
            render_devig_results(tool, ui, &palette);
        });
    });
}

fn render_header_controls(tool: &mut MarketDeviggingTool, ui: &mut egui::Ui) {
    ui.heading("Sharp Market Vig Removal & +EV Kelly Pricing");
    ui.label(
        "Compare Multiplicative, Additive, Power, and Shin devigging on multi-way sharp lines to filter phantom longshot EV.",
    );
    ui.horizontal_wrapped(|ui| {
        ui.label("Benchmark Scenarios:");
        if ui
            .button("2-Way Longshot (-500 / +350)")
            .accessible_hover_text("Load -500 / +350 sharp market with +400 soft underdog")
            .clicked()
        {
            tool.apply_two_way_longshot_preset();
        }
        if ui
            .button("3-Way Soccer (1.35 / 4.80 / 8.50)")
            .accessible_hover_text("Load 1X2 soccer market with 9.50 soft away line")
            .clicked()
        {
            tool.apply_three_way_soccer_preset();
        }
        if ui
            .button("6-Way Pod (101.0 Simplex Guard)")
            .accessible_hover_text("Load 6-way tournament pod where Additive raw prob hits -1.43%")
            .clicked()
        {
            tool.apply_six_way_pod_preset();
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Odds Format:");
        ui.radio_value(&mut tool.quote_format, QuoteFormat::Decimal, "Decimal");
        ui.radio_value(&mut tool.quote_format, QuoteFormat::American, "American");
        ui.separator();
        ui.label("Primary Devig Model:");
        ui.radio_value(&mut tool.method, DevigMethod::Multiplicative, "Multiplicative");
        ui.radio_value(&mut tool.method, DevigMethod::Additive, "Additive");
        ui.radio_value(&mut tool.method, DevigMethod::Power, "Power (Brent)");
        ui.radio_value(&mut tool.method, DevigMethod::Shin, "Shin (Brent)");
    });
}

fn render_quote_grid(tool: &mut MarketDeviggingTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("Outcomes ({}):", tool.decimal_quotes.len()));
        if ui.button("− Remove").clicked() && tool.decimal_quotes.len() > MIN_OUTCOMES {
            tool.decimal_quotes.pop();
            if tool.target_outcome >= tool.decimal_quotes.len() {
                tool.target_outcome = tool.decimal_quotes.len() - 1;
            }
        }
        if ui.button("+ Add Outcome").clicked() && tool.decimal_quotes.len() < MAX_GUI_OUTCOMES {
            tool.decimal_quotes.push(6.0);
        }
        ui.separator();
        ui.label("Target Outcome:");
        egui::ComboBox::from_id_salt("devig_target_outcome")
            .selected_text(format!("Outcome #{}", tool.target_outcome + 1))
            .show_ui(ui, |ui| {
                for idx in 0..tool.decimal_quotes.len() {
                    ui.selectable_value(
                        &mut tool.target_outcome,
                        idx,
                        format!("Outcome #{}", idx + 1),
                    );
                }
            });
        ui.label("Soft Offered Odds:");
        edit_odds_value(ui, &mut tool.offered_decimal_odds, tool.quote_format);
        ui.label("Bankroll ($):");
        ui.add(
            egui::DragValue::new(&mut tool.bankroll)
                .speed(25.0)
                .range(1.0..=1_000_000.0)
                .prefix("$"),
        );
    });
    ui.horizontal_wrapped(|ui| {
        for idx in 0..tool.decimal_quotes.len() {
            ui.group(|ui| {
                ui.label(format!("#{}", idx + 1));
                edit_odds_value(ui, &mut tool.decimal_quotes[idx], tool.quote_format);
            });
        }
    });
}

struct ComputedMarketArrays {
    n: usize,
    mult: [f64; MAX_MARKET_OUTCOMES],
    add_raw: [f64; MAX_MARKET_OUTCOMES],
    add_clamped: [f64; MAX_MARKET_OUTCOMES],
    power: [f64; MAX_MARKET_OUTCOMES],
    shin: [f64; MAX_MARKET_OUTCOMES],
    selected_meta: DevigMetadata,
    add_meta: DevigMetadata,
}

fn compute_all_arrays(
    odds: &[Odds],
    method: DevigMethod,
) -> Result<ComputedMarketArrays, String> {
    let n = odds.len();
    let mut mult = [0.0; MAX_MARKET_OUTCOMES];
    let mut add_raw = [0.0; MAX_MARKET_OUTCOMES];
    let mut add_clamped = [0.0; MAX_MARKET_OUTCOMES];
    let mut power = [0.0; MAX_MARKET_OUTCOMES];
    let mut shin = [0.0; MAX_MARKET_OUTCOMES];
    let mut sel_buf = [0.0; MAX_MARKET_OUTCOMES];

    devig_multiplicative_into(odds, &mut mult[..n]).map_err(|e| e.to_string())?;
    let add_meta = devig_additive_detailed_into(odds, &mut add_raw[..n], &mut add_clamped[..n])
        .map_err(|e| e.to_string())?;
    devig_power_into(odds, &mut power[..n]).map_err(|e| e.to_string())?;
    devig_shin_into(odds, &mut shin[..n]).map_err(|e| e.to_string())?;
    let selected_meta = devig_into(odds, method, &mut sel_buf[..n]).map_err(|e| e.to_string())?;

    Ok(ComputedMarketArrays {
        n,
        mult,
        add_raw,
        add_clamped,
        power,
        shin,
        selected_meta,
        add_meta,
    })
}

fn render_devig_results(
    tool: &MarketDeviggingTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let odds = match tool.build_odds_vec() {
        Ok(v) => v,
        Err(msg) => {
            ui.colored_label(palette.error, msg);
            return;
        }
    };
    let offered = match Odds::new(tool.offered_decimal_odds) {
        Ok(o) => o,
        Err(e) => {
            ui.colored_label(palette.error, format!("Invalid offered odds: {e}"));
            return;
        }
    };
    let arrays = match compute_all_arrays(&odds, tool.method) {
        Ok(a) => a,
        Err(msg) => {
            ui.colored_label(palette.error, msg);
            return;
        }
    };
    render_kpi_strip(&arrays.selected_meta, ui, palette);
    if let Some(diag) = arrays.add_meta.simplex_violation {
        render_simplex_banner(&diag, &arrays, ui, palette);
    }
    if let Ok(report) = compare_outcome_ev_all_methods(&odds, tool.target_outcome, offered) {
        render_ev_comparison(&report, tool.method, tool.bankroll, ui, palette);
    }
    ui.separator();
    render_outcome_table(&odds, &arrays, ui, palette);
    ui.separator();
    render_probability_chart(&odds, &arrays, ui, palette);
}

fn render_kpi_strip(meta: &DevigMetadata, ui: &mut egui::Ui, palette: &SemanticPalette) {
    let param_label = match meta.method {
        DevigMethod::Multiplicative => format!("Scale 1/S = {:.5}", meta.parameter),
        DevigMethod::Additive => format!("Shift/Outcome = {:.3}%", meta.parameter * 100.0),
        DevigMethod::Power => format!("Power Exponent k = {:.6}", meta.parameter),
        DevigMethod::Shin => format!("Insider Fraction z = {:.4}%", meta.parameter * 100.0),
    };
    ui.horizontal_wrapped(|ui| {
        ui.colored_label(
            palette.info,
            format!("Booksum: {:.2}%", meta.market.booksum * 100.0),
        );
        ui.separator();
        ui.label(format!("Overround: {:+.2}%", meta.market.overround * 100.0));
        ui.separator();
        ui.label(format!(
            "Turnover Margin: {:.2}%",
            meta.market.turnover_margin * 100.0
        ));
        ui.separator();
        ui.colored_label(palette.accent, param_label);
        if meta.solver_iterations > 0 {
            ui.label(format!("({} Brent evals)", meta.solver_iterations));
        }
    });
}

fn render_simplex_banner(
    diag: &math_explorer::pure_math::statistics::market_pricing::SimplexViolationDiagnostic,
    arrays: &ComputedMarketArrays,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    egui::Frame::group(ui.style())
        .fill(palette.error_banner_bg)
        .stroke(egui::Stroke::new(1.5_f32, palette.error_banner_stroke))
        .show(ui, |ui| {
            ui.colored_label(
                palette.warning,
                format!(
                    "⚠ Additive Simplex Boundary Violation: {} longshot outcome(s) went non-positive (worst: Outcome #{} raw = {:.2}%, equal shift = {:.2}%). Clamped to 0.00% and renormalized onto the simplex.",
                    diag.violated_count,
                    diag.worst_outcome_index + 1,
                    diag.min_raw_probability * 100.0,
                    diag.margin_per_outcome * 100.0
                ),
            );
            let idx = diag.worst_outcome_index;
            ui.label(format!(
                "Outcome #{} Comparison -> Raw Additive: {:.2}% | Clamped Simplex: {:.2}% | Power: {:.2}% | Shin: {:.2}%",
                idx + 1,
                arrays.add_raw[idx] * 100.0,
                arrays.add_clamped[idx] * 100.0,
                arrays.power[idx] * 100.0,
                arrays.shin[idx] * 100.0
            ));
        });
}

fn render_ev_comparison(
    report: &AllMethodsEvReport,
    selected: DevigMethod,
    bankroll: f64,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let active = report.for_method(selected);
    let status_color = if active.has_edge {
        palette.success
    } else {
        palette.error
    };
    let (fk_amt, hk_amt, qk_amt) = active.recommended_stakes(bankroll).unwrap_or((0.0, 0.0, 0.0));
    ui.horizontal_wrapped(|ui| {
        ui.colored_label(
            status_color,
            format!(
                "Outcome #{} @ {:.2} ({:?}): EV = {:+.2}% | Fair Odds = {:.2}",
                report.outcome_index + 1,
                active.offered_odds.value(),
                selected,
                active.ev_percent,
                active.fair_decimal_odds
            ),
        );
        ui.separator();
        ui.label(format!(
            "Full Kelly: {:.2}% (${:.2}) | Half: ${:.2} | Quarter: ${:.2}",
            active.full_kelly.value() * 100.0,
            fk_amt,
            hk_amt,
            qk_amt
        ));
    });
    ui.horizontal_wrapped(|ui| {
        ui.colored_label(palette.muted_text, "Cross-Model EV Check:");
        for (label, m) in [
            ("Multiplicative", DevigMethod::Multiplicative),
            ("Additive", DevigMethod::Additive),
            ("Power", DevigMethod::Power),
            ("Shin", DevigMethod::Shin),
        ] {
            let ev = report.for_method(m).ev_percent;
            let col = if ev > 0.0 { palette.success } else { palette.muted_text };
            ui.colored_label(col, format!("{label}: {ev:+.2}%"));
        }
    });
}

fn render_outcome_table(
    odds: &[Odds],
    arrays: &ComputedMarketArrays,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    egui::Grid::new("market_devig_outcome_table")
        .striped(true)
        .spacing([18.0, 6.0])
        .show(ui, |ui| {
            for col in [
                "Outcome",
                "Quoted",
                "Implied",
                "Multiplicative",
                "Additive (Raw)",
                "Additive (Clamped)",
                "Power",
                "Shin",
                "Shin Fair Odds",
            ] {
                ui.strong(col);
            }
            ui.end_row();
            for i in 0..arrays.n {
                ui.label(format!("#{}", i + 1));
                ui.label(format!("{:.2}", odds[i].value()));
                ui.label(format!("{:.2}%", odds[i].implied_probability() * 100.0));
                ui.label(format!("{:.2}%", arrays.mult[i] * 100.0));
                let raw_pct = arrays.add_raw[i] * 100.0;
                if raw_pct <= 0.0 {
                    ui.colored_label(palette.error, format!("{raw_pct:.2}%"));
                } else {
                    ui.label(format!("{raw_pct:.2}%"));
                }
                ui.label(format!("{:.2}%", arrays.add_clamped[i] * 100.0));
                ui.label(format!("{:.2}%", arrays.power[i] * 100.0));
                ui.label(format!("{:.2}%", arrays.shin[i] * 100.0));
                let shin_fair = if arrays.shin[i] > 0.0 {
                    format!("{:.2}", 1.0 / arrays.shin[i])
                } else {
                    "∞".to_string()
                };
                ui.label(shin_fair);
                ui.end_row();
            }
        });
}

fn render_probability_chart(
    odds: &[Odds],
    arrays: &ComputedMarketArrays,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let mut implied_bars = Vec::with_capacity(arrays.n);
    let mut mult_bars = Vec::with_capacity(arrays.n);
    let mut power_bars = Vec::with_capacity(arrays.n);
    let mut shin_bars = Vec::with_capacity(arrays.n);
    for i in 0..arrays.n {
        let x = (i + 1) as f64;
        implied_bars.push(Bar::new(x - 0.24, odds[i].implied_probability() * 100.0).width(0.15));
        mult_bars.push(Bar::new(x - 0.08, arrays.mult[i] * 100.0).width(0.15));
        power_bars.push(Bar::new(x + 0.08, arrays.power[i] * 100.0).width(0.15));
        shin_bars.push(Bar::new(x + 0.24, arrays.shin[i] * 100.0).width(0.15));
    }
    Plot::new("market_devig_prob_chart")
        .height(230.0)
        .legend(Legend::default())
        .x_axis_label("Outcome Index")
        .y_axis_label("Probability (%)")
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new("Raw Implied %", implied_bars).color(palette.series[0]));
            plot_ui.bar_chart(BarChart::new("Multiplicative %", mult_bars).color(palette.series[1]));
            plot_ui.bar_chart(BarChart::new("Power %", power_bars).color(palette.series[2]));
            plot_ui.bar_chart(BarChart::new("Shin %", shin_bars).color(palette.series[4]));
        });
}

// [cite:clinical_trials_statistics]
