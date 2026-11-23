#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::SemanticPalette;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use math_explorer::epidemiology::analytics::calculate_final_size;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState, heterogeneous_r0};
use math_explorer::epidemiology::stochastic::probability_of_extinction;

/// Live compartmental counts and residual analytical outbreak indicators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NetworkAnalyticsSummary {
    pub susceptible: usize,
    pub infected: usize,
    pub recovered: usize,
    pub vaccinated: usize,
    pub mean_active_degree: f64,
    pub active_degree_variance: f64,
    pub effective_r0: f64,
    pub extinction_probability: f64,
    pub expected_additional_infections: f64,
}

/// Evaluates residual `heterogeneous_r0`, `probability_of_extinction`, and guarded `calculate_final_size`.
pub(crate) fn compute_network_analytics(model: &NetworkEpidemicModel) -> NetworkAnalyticsSummary {
    let mut susceptible = 0_usize;
    let mut infected = 0_usize;
    let mut recovered = 0_usize;
    let mut vaccinated = 0_usize;

    for &state in model.states().iter().take(model.num_nodes()) {
        match state {
            NodeState::Susceptible => susceptible += 1,
            NodeState::Infected => infected += 1,
            NodeState::Recovered => recovered += 1,
            NodeState::Vaccinated => vaccinated += 1,
        }
    }

    let (mean_k, var_k) = model.active_degree_moments();
    let raw_r0 = heterogeneous_r0(model.beta(), model.gamma(), mean_k, var_k);
    let effective_r0 = if raw_r0.is_finite() && raw_r0 > 0.0 {
        raw_r0
    } else {
        0.0
    };

    let extinction_probability = if infected == 0 {
        1.0
    } else {
        probability_of_extinction(effective_r0, infected as f64).clamp(0.0, 1.0)
    };

    let s0 = susceptible as f64;
    let n = model.num_nodes() as f64;
    let expected_additional_infections = if effective_r0 <= 0.0 || s0 <= 0.0 || n <= 0.0 {
        0.0
    } else {
        match calculate_final_size(effective_r0, s0, n) {
            Ok(s_inf) if s_inf.is_finite() => (s0 - s_inf).clamp(0.0, s0),
            _ => 0.0,
        }
    };

    NetworkAnalyticsSummary {
        susceptible,
        infected,
        recovered,
        vaccinated,
        mean_active_degree: mean_k,
        active_degree_variance: var_k,
        effective_r0,
        extinction_probability,
        expected_additional_infections,
    }
}

pub(super) fn render_compartment_and_analytics_panel(
    ui: &mut egui::Ui,
    summary: &NetworkAnalyticsSummary,
    used_budget: usize,
    budget: usize,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let remaining = budget.saturating_sub(used_budget);
    let budget_color = if remaining > 0 {
        palette.success
    } else {
        palette.warning
    };

    ui.label(
        egui::RichText::new(format!(
            "Budget Used: {used_budget} / {budget} ({remaining} remaining)"
        ))
        .color(budget_color)
        .strong(),
    );

    ui.horizontal_wrapped(|ui| {
        ui.label(
            egui::RichText::new(format!("Susceptible: {}", summary.susceptible))
                .color(palette.axis_z),
        );
        ui.label(
            egui::RichText::new(format!("Infected: {}", summary.infected)).color(palette.axis_x),
        );
        ui.label(
            egui::RichText::new(format!("Recovered: {}", summary.recovered)).color(palette.axis_y),
        );
        ui.label(
            egui::RichText::new(format!("Vaccinated: {}", summary.vaccinated))
                .color(palette.accent),
        );
    });

    ui.separator();
    ui.strong("Live Domain Analytics");
    render_copyable_metric(
        ui,
        "Effective R₀",
        &format!("{:.2}", summary.effective_r0),
    );
    render_copyable_metric(
        ui,
        "Extinction Probability",
        &format!("{:.1}%", summary.extinction_probability * 100.0),
    );
    render_copyable_metric(
        ui,
        "Expected Additional Infections",
        &format!("{:.1}", summary.expected_additional_infections),
    );
}
