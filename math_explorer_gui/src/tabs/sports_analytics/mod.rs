#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::tabs::{ExplorerTab, TabDescriptor};
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotUi, VLine};
use math_explorer::sports_quant::{
    expected_value, kelly_fraction, kelly_variants, Odds, UnitInterval,
};

pub mod clv_ruin;
pub mod live_momentum;
pub mod market_devig;
pub mod match_heatmap;
pub mod presets;
pub mod ratings;
pub mod sgp_copula;
pub mod zip_props;

#[cfg(test)]
mod tests;

/// Interactive Sports Analytics tab under Applied Math hosting quantitative pricing,
/// dependence, count, in-play SDE, rating, and Kelly risk tools.
pub struct SportsAnalyticsTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for SportsAnalyticsTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("sports_analytics"),
        }
    }
}

pub(super) fn render_tool_header(ui: &mut egui::Ui, tool_name: &str) {
    ui.horizontal(|ui| {
        ui.heading(tool_name);
        let help_btn = ui
            .add(egui::Button::new("❓").small())
            .accessible_hover_text(format!("Toggle Theory Context Portal for {tool_name}"));
        if help_btn.clicked() {
            let ctx = ui.ctx();
            let next = !ctx.data(|d| {
                d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL"))
                    .unwrap_or(false)
            });
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), next);
                if let Some(inst_key) =
                    d.get_temp::<egui::Id>(egui::Id::new("CURRENT_THEORY_PORTAL_INST_ID"))
                {
                    d.insert_temp(inst_key, next);
                }
            });
        }
    });
    ui.separator();
}

/// Evaluates expected value per dollar, full-Kelly percentage, and half-Kelly dollar stake
/// directly from a `UnitInterval` model probability and decimal bookmaker odds.
pub(super) fn evaluate_kelly_sizing(
    fair_prob: UnitInterval,
    decimal_odds: f64,
    bankroll: f64,
) -> (f64, f64, f64) {
    let Ok(odds) = Odds::new(decimal_odds) else {
        return (0.0, 0.0, 0.0);
    };
    let ev = expected_value(&fair_prob, &odds);
    let full_pct = kelly_fraction(&fair_prob, &odds)
        .map(|f| f.value() * 100.0)
        .unwrap_or(0.0);
    let half_stake = kelly_variants::half_kelly(&fair_prob, &odds)
        .and_then(|f| f.bet_amount(bankroll))
        .unwrap_or(0.0);
    (ev, full_pct, half_stake)
}

pub(super) fn render_sports_left_panel(
    parent_ui: &mut egui::Ui,
    panel_salt: &'static str,
    title: &'static str,
    controls: &mut dyn FnMut(&mut egui::Ui),
) {
    let side_id = crate::framework::scoped_panel_id(parent_ui, panel_salt);
    crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
        render_tool_header(ui, title);
        controls(ui);
    });
}

pub(super) fn render_sports_central_panel(
    parent_ui: &mut egui::Ui,
    central: &mut dyn FnMut(&mut egui::Ui, &SemanticPalette),
) {
    egui::CentralPanel::default().show_inside(parent_ui, |ui| {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        central(ui, &palette);
    });
}

pub(super) fn render_sports_plot(
    ui: &mut egui::Ui,
    id_salt: &'static str,
    x_label: &'static str,
    y_label: &'static str,
    draw: &mut dyn FnMut(&mut PlotUi<'_>),
) {
    Plot::new(ui.id().with(id_salt))
        .legend(Legend::default())
        .x_axis_label(x_label)
        .y_axis_label(y_label)
        .show(ui, |plot_ui| draw(plot_ui));
}

pub(super) fn slider_f64(
    ui: &mut egui::Ui,
    val: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    text: &'static str,
) -> egui::Response {
    ui.add(egui::Slider::new(val, range).text(text))
}

pub(super) fn slider_f64_step(
    ui: &mut egui::Ui,
    val: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    step: f64,
    text: &'static str,
) -> egui::Response {
    ui.add(egui::Slider::new(val, range).step_by(step).text(text))
}

pub(super) fn plot_line(
    name: &str,
    pts: &[[f64; 2]],
    color: egui::Color32,
    width: f32,
    dashed: Option<f32>,
) -> Line<'static> {
    let mut line = Line::new(name.to_owned(), pts.to_vec())
        .color(color)
        .width(width);
    if let Some(length) = dashed {
        line = line.style(LineStyle::Dashed { length });
    }
    line
}

pub(super) fn plot_vline(name: &'static str, x: f64, color: egui::Color32, dashed: f32) -> VLine {
    VLine::new(name, x)
        .color(color)
        .style(LineStyle::Dashed { length: dashed })
}

pub(super) fn plot_hline(
    name: &'static str,
    y: f64,
    color: egui::Color32,
    dashed: Option<f32>,
) -> HLine {
    let mut hline = HLine::new(name, y).color(color);
    if let Some(length) = dashed {
        hline = hline.style(LineStyle::Dashed { length });
    }
    hline
}

pub(super) fn parse_state_value(raw: &str) -> Result<serde_json::Value, String> {
    serde_json::from_str(raw).map_err(|e| e.to_string())
}

impl ExplorerTab for SportsAnalyticsTab {
    fn name(&self) -> &'static str {
        "Sports Analytics"
    }

    fn academic_domain(&self) -> &'static str {
        "Applied Math"
    }

    fn url_slug(&self) -> &'static str {
        "sports_analytics"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "sports_analytics");
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

inventory::submit! {
    TabDescriptor {
        id: "sports_analytics",
        order: 37,
        build: || Box::new(SportsAnalyticsTab::default()),
    }
}

// [cite:sports_quantitative_modeling]
