#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls and factorization diagnostics for [`super::state::ConditioningWorkbenchTool`].

use super::state::{
    ConditioningWorkbenchTool, MatrixFamily, MAX_COLS, MAX_ROWS, MIN_COLS, MIN_ROWS,
};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;

impl ConditioningWorkbenchTool {
    pub(crate) fn select_family(&mut self, family: MatrixFamily) {
        self.config.family = family;
        match family {
            MatrixFamily::Lauchli => {
                self.config.rows = 7;
                self.config.cols = 6;
                self.config.perturbation_log10 = -8.0;
                self.config.noise_std = 0.0;
                self.config.ridge_lambda_log10 = -6.0;
            }
            MatrixFamily::Vandermonde => {
                self.config.rows = 16;
                self.config.cols = 10;
                self.config.noise_std = 0.0;
                self.config.ridge_lambda_log10 = -8.0;
            }
            MatrixFamily::Hilbert => {
                self.config.rows = 8;
                self.config.cols = 8;
                self.config.noise_std = 1e-4;
                self.config.ridge_lambda_log10 = -6.0;
            }
            MatrixFamily::RankDeficient => {
                self.config.rows = 12;
                self.config.cols = 6;
                self.config.perturbation_log10 = -11.0;
                self.config.noise_std = 0.02;
                self.config.ridge_lambda_log10 = -4.0;
            }
        }
        self.config.truncation_rank = (self.config.cols / 2).clamp(1, MAX_COLS);
        self.recompute();
    }

    fn show_family_selector(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Test Matrix Family").strong());
        ui.horizontal_wrapped(|ui| {
            for family in [
                MatrixFamily::Lauchli,
                MatrixFamily::Vandermonde,
                MatrixFamily::Hilbert,
                MatrixFamily::RankDeficient,
            ] {
                let selected = self.config.family == family;
                if ui
                    .selectable_label(selected, family.label())
                    .accessible_hover_text(format!("Load {} matrix testbed", family.label()))
                    .clicked()
                {
                    self.select_family(family);
                }
            }
        });
    }

    fn show_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Dimensions & Conditioning").strong());
        let mut changed = false;

        let mut rows = self.config.rows;
        if ui
            .add(egui::Slider::new(&mut rows, MIN_ROWS..=MAX_ROWS).text("Rows m"))
            .accessible_hover_text("Bounded row count m in [3, 32]")
            .changed()
        {
            self.config.rows = rows.clamp(MIN_ROWS, MAX_ROWS);
            changed = true;
        }

        let mut cols = self.config.cols;
        if ui
            .add(egui::Slider::new(&mut cols, MIN_COLS..=MAX_COLS).text("Cols n"))
            .accessible_hover_text("Bounded predictor column count n in [2, 16]")
            .changed()
        {
            self.config.cols = cols.clamp(MIN_COLS, MAX_COLS);
            changed = true;
        }

        let mut mu_log = self.config.perturbation_log10;
        if ui
            .add(egui::Slider::new(&mut mu_log, -12.0..=-1.0).text("log10(μ)"))
            .accessible_hover_text("Log10 diagonal perturbation controlling singular value gap")
            .changed()
        {
            self.config.perturbation_log10 = mu_log.clamp(-12.0, -1.0);
            changed = true;
        }

        let mut noise = self.config.noise_std;
        if ui
            .add(egui::Slider::new(&mut noise, 0.0..=1.0).text("Noise σ"))
            .accessible_hover_text("Observation perturbation amplitude added to b = A x* + noise")
            .changed()
        {
            self.config.noise_std = noise.clamp(0.0, 1.0);
            changed = true;
        }

        let mut ridge_log = self.config.ridge_lambda_log10;
        if ui
            .add(egui::Slider::new(&mut ridge_log, -14.0..=2.0).text("log10(λ Ridge)"))
            .accessible_hover_text("Log10 Tikhonov regularization penalty for Augmented-QR Ridge")
            .changed()
        {
            self.config.ridge_lambda_log10 = ridge_log.clamp(-14.0, 2.0);
            changed = true;
        }

        let max_k = self.config.rows.min(self.config.cols).clamp(1, MAX_COLS);
        let mut rank_k = self.config.truncation_rank.clamp(1, max_k);
        if ui
            .add(egui::Slider::new(&mut rank_k, 1..=max_k).text("SVD Rank k"))
            .accessible_hover_text("Target rank k for low-rank SVD approximation error")
            .changed()
        {
            self.config.truncation_rank = rank_k.clamp(1, MAX_COLS);
            changed = true;
        }

        if changed {
            self.recompute();
        }
    }

    fn show_invariants_card(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let s = &self.snapshot;
        ui.group(|ui| {
            ui.label(egui::RichText::new("Factorization & Norm Diagnostics").strong());
            let cond_str = if s.condition_number.is_finite() {
                format!("{:.3e}", s.condition_number)
            } else {
                "∞ (rank-deficient)".to_string()
            };
            let cond_color = if !s.condition_number.is_finite() || s.condition_number > 1e8 {
                palette.warning
            } else {
                palette.success
            };
            ui.colored_label(cond_color, format!("Condition κ₂(A): {cond_str}"));
            ui.label(format!(
                "Effective Rank: {} / {}",
                s.effective_rank,
                self.config.rows.min(self.config.cols)
            ));
            ui.label(format!(
                "‖A‖₁ = {:.3}   ‖A‖₂ = {:.3}",
                s.norms[0], s.norms[1]
            ));
            ui.label(format!(
                "‖A‖∞ = {:.3}   ‖A‖_F = {:.3}",
                s.norms[2], s.norms[3]
            ));
            ui.label(format!("‖Q R - A‖_F = {:.2e}", s.qr_recon_error));
            ui.label(format!(
                "‖A - A_k‖_F (k={}) = {:.3e}",
                self.config
                    .truncation_rank
                    .min(self.config.rows.min(self.config.cols)),
                s.low_rank_error
            ));
            let (chol_txt, chol_col) = if s.cholesky_spd {
                ("Cholesky L L^T: SPD Verified", palette.success)
            } else {
                (
                    "Cholesky L L^T: Not Square SPD",
                    ui.visuals().weak_text_color(),
                )
            };
            ui.colored_label(chol_col, chol_txt);
        });
    }

    pub(crate) fn show_controls_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Conditioning Workbench");
        ui.separator();
        self.show_family_selector(ui);
        ui.separator();
        self.show_parameter_sliders(ui);
        ui.separator();
        self.show_invariants_card(ui);
    }
}

// [cite:advanced_linear_algebra]
