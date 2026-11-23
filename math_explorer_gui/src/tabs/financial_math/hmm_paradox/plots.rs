//! 2D branching trellis and timeline posterior comparison plots for `HiddenMarkovParadoxTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HiddenMarkovParadoxTool, NUM_HMM_STATES, STATE_NAMES};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, Points, VLine};

/// Renders the 2D `(time t, state i)` branching trellis contrasting Viterbi and forbidden Marginal paths.
pub fn render_branching_trellis_plot(
    tool: &mut HiddenMarkovParadoxTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let t_len = tool.config.observations.len();
    let viterbi_pts: Vec<[f64; 2]> = tool
        .baseline
        .viterbi_path
        .iter()
        .enumerate()
        .map(|(t, &s)| [t as f64, s as f64])
        .collect();
    let map_pts: Vec<[f64; 2]> = tool
        .baseline
        .marginal_path
        .iter()
        .enumerate()
        .map(|(t, &s)| [t as f64, s as f64])
        .collect();
    let branch_viterbi_pts: Vec<[f64; 2]> = tool
        .branched
        .viterbi_path
        .iter()
        .enumerate()
        .map(|(t, &s)| [t as f64, (s as f64) + 0.07])
        .collect();
    let fork_step_x = tool.config.fork_step as f64;

    Plot::new("hmm_branching_trellis_plot")
        .height(270.0)
        .allow_scroll(false)
        .x_axis_label("Time Step t")
        .y_axis_label("Hidden State (0=Calm, 1=Shock, 2=Bridge)")
        .include_x(-0.2)
        .include_x((t_len as f64) - 0.8)
        .include_y(-0.35)
        .include_y(2.35)
        .show(ui, |plot_ui| {
            draw_trellis_lattice_edges(tool, plot_ui, palette, t_len);
            plot_ui.vline(
                VLine::new("Time-Travel Fork Step", fork_step_x)
                    .color(palette.accent)
                    .style(LineStyle::Dotted { spacing: 5.0_f32 }),
            );
            plot_ui.line(
                Line::new("Baseline Viterbi Joint Path", viterbi_pts)
                    .color(palette.success)
                    .width(3.2_f32),
            );
            plot_ui.line(
                Line::new("Baseline Marginal MAP Path", map_pts)
                    .color(palette.error)
                    .style(LineStyle::Dashed { length: 7.0_f32 })
                    .width(2.6_f32),
            );
            plot_ui.line(
                Line::new("Forked Timeline Viterbi Path", branch_viterbi_pts)
                    .color(palette.info)
                    .style(LineStyle::Dotted { spacing: 4.5_f32 })
                    .width(2.4_f32),
            );
            draw_posterior_trellis_nodes(tool, plot_ui, palette, t_len);
            tool.callout_manager.render_on_plot(plot_ui);
        });
}

fn draw_trellis_lattice_edges(
    tool: &HiddenMarkovParadoxTool,
    plot_ui: &mut egui_plot::PlotUi,
    palette: &SemanticPalette,
    t_len: usize,
) {
    let trans = tool.hmm.transitions();
    for t in 0..t_len.saturating_sub(1) {
        for i in 0..NUM_HMM_STATES {
            for j in 0..NUM_HMM_STATES {
                if trans[(i, j)] > 0.0 {
                    plot_ui.line(
                        Line::new("", vec![[t as f64, i as f64], [(t + 1) as f64, j as f64]])
                            .color(palette.graph_edge)
                            .width(1.0_f32),
                    );
                }
            }
        }
    }
}

fn draw_posterior_trellis_nodes(
    tool: &HiddenMarkovParadoxTool,
    plot_ui: &mut egui_plot::PlotUi,
    palette: &SemanticPalette,
    t_len: usize,
) {
    for (state, &state_name) in STATE_NAMES.iter().enumerate().take(NUM_HMM_STATES) {
        for t in 0..t_len {
            let g = tool.baseline_scratch.gamma_at(state, t);
            let radius = (4.0 + 9.0 * g) as f32;
            plot_ui.points(
                Points::new(state_name, vec![[t as f64, state as f64]])
                    .radius(radius)
                    .color(palette.series_color(state)),
            );
        }
    }
}

/// Renders smoothed Shock-state posterior $\gamma_t(1)$ and step scaling across Baseline vs Forked timelines.
pub fn render_posterior_comparison_plot(
    tool: &HiddenMarkovParadoxTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let t_len = tool.config.observations.len();
    let base_shock: Vec<[f64; 2]> = (0..t_len)
        .map(|t| [t as f64, tool.baseline_scratch.gamma_at(1, t)])
        .collect();
    let base_bridge: Vec<[f64; 2]> = (0..t_len)
        .map(|t| [t as f64, tool.baseline_scratch.gamma_at(2, t)])
        .collect();
    let branch_shock: Vec<[f64; 2]> = (0..t_len)
        .map(|t| [t as f64, tool.branched_scratch.gamma_at(1, t)])
        .collect();
    let branch_scale: Vec<[f64; 2]> = tool
        .branched_scratch
        .scaling_slice(t_len)
        .iter()
        .enumerate()
        .map(|(t, &c)| [t as f64, c])
        .collect();

    Plot::new("hmm_posterior_timeline_comparison_plot")
        .height(270.0)
        .allow_scroll(false)
        .x_axis_label("Time Step t")
        .y_axis_label("Smoothed Posterior γ_t(i) / Predictive Scale c_t")
        .include_x(0.0)
        .include_x((t_len.saturating_sub(1)) as f64)
        .include_y(0.0)
        .include_y(1.02)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Baseline γ_t(Shock=1)", base_shock)
                    .color(palette.error)
                    .width(2.6_f32),
            );
            plot_ui.line(
                Line::new("Baseline γ_t(Bridge=2)", base_bridge)
                    .color(palette.warning)
                    .width(2.2_f32),
            );
            plot_ui.line(
                Line::new("Forked γ'_t(Shock=1)", branch_shock)
                    .color(palette.info)
                    .style(LineStyle::Dashed { length: 6.5_f32 })
                    .width(2.5_f32),
            );
            plot_ui.line(
                Line::new("Forked Predictive Scale c'_t", branch_scale)
                    .color(palette.success)
                    .style(LineStyle::Dotted { spacing: 5.0_f32 })
                    .width(2.0_f32),
            );
        });
}

// [cite:clinical_trials_statistics]
