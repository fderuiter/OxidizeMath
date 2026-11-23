//! User interface controls, time-travel observation fork editor, and KPI cards for `HiddenMarkovParadoxTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::challenge::CHALLENGE_HMM_FORBIDDEN_FORK;
use super::plots::{render_branching_trellis_plot, render_posterior_comparison_plot};
use super::{HiddenMarkovParadoxTool, MAX_HMM_STEPS, NUM_HMM_OBS, OBS_NAMES};
use crate::app::SemanticPalette;
use eframe::egui;

/// Renders the complete `HiddenMarkovParadoxTool` interactive workbench.
pub fn show_hmm_paradox_tool(tool: &mut HiddenMarkovParadoxTool, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());

    ui.heading("HMM Forbidden-Path Trellis & Time-Travel Branching Lab");
    ui.label(
        "Inspect why pointwise posterior smoothing γ_t(i) can pick a zero-probability state path across A[1,1]=0 while Viterbi recovers the valid joint path.",
    );
    ui.add_space(4.0);

    render_kpi_cards(tool, ui, &palette);
    ui.separator();

    render_controls(tool, ui, &palette);
    ui.separator();

    ui.columns(2, |cols| {
        cols[0].strong("2D Hidden-State Trellis: Viterbi vs Marginal MAP & Fork");
        cols[0].small("Node size ∝ γ_t(i); missing 1→1 edge marks forbidden A[1,1] = 0 transition");
        render_branching_trellis_plot(tool, &mut cols[0], &palette);

        cols[1].strong("Smoothed Posterior γ_t(i) & Time-Travel Branch Comparison");
        cols[1].small(
            "Forking token y_t propagates both forward (filtering) and backward (smoothing)",
        );
        render_posterior_comparison_plot(tool, &mut cols[1], &palette);
    });
}

fn render_kpi_cards(tool: &HiddenMarkovParadoxTool, ui: &mut egui::Ui, palette: &SemanticPalette) {
    let b = &tool.baseline;
    ui.horizontal_wrapped(|ui| {
        ui.group(|ui| {
            ui.label("Forward Likelihood P(Y)");
            ui.colored_label(
                palette.info,
                format!(
                    "P(Y) = {:.7} (ln L = {:.3})",
                    b.forward_likelihood, b.forward_log_likelihood
                ),
            );
            ui.small(format!(
                "Filter P(X_T | Y) = [{:.2}, {:.2}, {:.2}]",
                b.final_filter_belief[0], b.final_filter_belief[1], b.final_filter_belief[2]
            ));
        });
        render_trellis_and_fork_cards(tool, ui, palette);
    });

    if b.has_forbidden_marginal_step {
        ui.colored_label(
            palette.warning,
            format!(
                "⚡ Pointwise Marginal Paradox: Independent smoothing argmax selects {:?} with joint probability P = 0.0 (violates A[1,1] = 0), whereas Viterbi routes through Bridge state {:?} (P = {:.7}).",
                b.marginal_path, b.viterbi_path, b.viterbi_joint_prob
            ),
        );
    }
}

fn render_trellis_and_fork_cards(
    tool: &HiddenMarkovParadoxTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let b = &tool.baseline;
    let f = &tool.branched;
    let map_col = if b.has_forbidden_marginal_step {
        palette.error
    } else {
        palette.success
    };
    ui.group(|ui| {
        ui.label("Pointwise Marginal MAP vs Viterbi");
        ui.colored_label(
            map_col,
            format!(
                "MAP {:?} (P = {:.6})",
                b.marginal_path, b.marginal_joint_prob
            ),
        );
        ui.small(format!(
            "Viterbi {:?} (P = {:.7})",
            b.viterbi_path, b.viterbi_joint_prob
        ));
    });

    ui.group(|ui| {
        ui.label(format!(
            "Forked Timeline (t={} → {})",
            tool.config.fork_step, OBS_NAMES[tool.config.fork_token]
        ));
        ui.colored_label(
            palette.accent,
            format!(
                "P(Y') = {:.6} | Viterbi {:?}",
                f.forward_likelihood, f.viterbi_path
            ),
        );
        ui.small(format!(
            "Fork MAP {:?} (P = {:.6})",
            f.marginal_path, f.marginal_joint_prob
        ));
    });
}

fn render_controls(
    tool: &mut HiddenMarkovParadoxTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
) {
    let locked = tool.active_challenge == Some(CHALLENGE_HMM_FORBIDDEN_FORK);
    let mut changed = false;

    ui.columns(2, |cols| {
        let max_step = tool.config.observations.len().saturating_sub(1);
        changed |= cols[0]
            .add(
                egui::Slider::new(&mut tool.config.fork_step, 0..=max_step)
                    .text("Time-Travel Fork Step t"),
            )
            .changed();
        cols[0].horizontal(|ui| {
            ui.label("Fork Token y'_t:");
            for (tok, &tok_name) in OBS_NAMES.iter().enumerate().take(NUM_HMM_OBS) {
                if ui
                    .selectable_label(tool.config.fork_token == tok, tok_name)
                    .clicked()
                {
                    tool.config.fork_token = tok;
                    changed = true;
                }
            }
        });
        changed |= cols[0]
            .add(
                egui::Slider::new(&mut tool.config.bridge_to_shock_prob, 0.10..=0.95)
                    .text("Bridge→Shock A[2,1]"),
            )
            .changed();
        changed |= cols[0]
            .add(
                egui::Slider::new(&mut tool.config.shock_emission_focus, 0.30..=0.90)
                    .text("Shock Spike Emission B[1,1]"),
            )
            .changed();

        cols[1].add_enabled_ui(!locked, |ui| {
            changed |= ui
                .add(
                    egui::Slider::new(&mut tool.config.shock_self_loop_prob, 0.0..=0.60)
                        .text("Shock Self-Loop A[1,1]"),
                )
                .changed();
            changed |= render_sequence_editor(&mut tool.config.observations, ui);
        });
    });

    render_hmm_action_buttons(tool, ui, palette, locked);

    if changed {
        tool.recalculate();
    }
}

fn render_hmm_action_buttons(
    tool: &mut HiddenMarkovParadoxTool,
    ui: &mut egui::Ui,
    palette: &SemanticPalette,
    locked: bool,
) {
    ui.horizontal(|ui| {
        if ui.button("Starter 3-State Paradox Preset").clicked() {
            tool.reset_starter_preset();
        }
        ui.add_enabled_ui(!locked, |ui| {
            if ui.button("Simulate Rollout (Seeded RNG)").clicked() {
                tool.simulate_seeded_rollout();
            }
        });
        if locked {
            ui.colored_label(
                palette.info,
                "🔒 Challenge locks active on baseline Y = [0, 1, 1] & forbidden A[1,1] = 0.0",
            );
        }
    });
}

fn render_sequence_editor(observations: &mut Vec<usize>, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.label("Baseline Tokens Y:");
        for (idx, tok) in observations.iter_mut().enumerate() {
            if ui.button(format!("t{idx}:{tok}")).clicked() {
                *tok = (*tok + 1) % NUM_HMM_OBS;
                changed = true;
            }
        }
        if observations.len() < MAX_HMM_STEPS && ui.small_button("+ Step").clicked() {
            observations.push(1);
            changed = true;
        }
        if observations.len() > 2 && ui.small_button("- Step").clicked() {
            observations.pop();
            changed = true;
        }
    });
    changed
}

// [cite:clinical_trials_statistics]
