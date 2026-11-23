//! Constrained IMRT solver (`L-BFGS-B`) and Tumor-vs-OAR Pareto frontier controller for [`super::dose::DoseCalculationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::imrt::{
    ImrtWorkspace, IMRT_BEAM_COUNT, IMRT_COST_ALPHA, IMRT_COST_BETA_OAR, IMRT_OAR_CENTER,
    IMRT_OAR_RADIUS_CM, IMRT_PHANTOM_SIZE_CM, IMRT_TUMOR_CENTER, IMRT_TUMOR_RADIUS_CM,
    IMRT_VOXEL_COUNT,
};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::app::SemanticPalette;
use crate::framework::CalloutManager;
use eframe::egui::{self, Color32};
use egui_plot::{Line, MarkerShape, Plot, PlotImage, PlotPoint, PlotUi, Points, Polygon};
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::physics::medical::optimization::{
    compute_imrt_pareto_frontier, solve_imrt_weights_in_place, ImrtParetoPoint, ImrtPlanConfig,
    ImrtSolverWorkspace,
};

/// Maximum beam weight permitted by the GUI slider (`[0.0, 6.0]`).
pub const IMRT_MAX_BEAM_WEIGHT: f64 = 6.0;
/// Number of weighted-sum samples for the IMRT Tumor-vs-OAR Pareto trade-off curve.
pub const IMRT_PARETO_SAMPLES: usize = 9;

/// Preallocated controller for box-constrained IMRT solving and Pareto frontier exploration.
pub struct ImrtParetoController {
    influence_matrix: DMatrix<f64>,
    solver_ws: ImrtSolverWorkspace,
    weights_vec: DVector<f64>,
    /// Cached non-dominated Tumor-vs-OAR trade-off points.
    pub pareto_points: Vec<ImrtParetoPoint>,
    /// Whether the split-view Pareto trade-off plot is displayed alongside the phantom heatmap.
    pub show_pareto_curve: bool,
    /// Active Pareto operating point index selected by the user.
    pub selected_idx: usize,
}

impl Default for ImrtParetoController {
    fn default() -> Self {
        Self {
            influence_matrix: DMatrix::zeros(IMRT_VOXEL_COUNT, IMRT_BEAM_COUNT),
            solver_ws: ImrtSolverWorkspace::new(
                IMRT_VOXEL_COUNT,
                IMRT_BEAM_COUNT,
                IMRT_MAX_BEAM_WEIGHT,
                80,
            )
            .expect("valid IMRT solver workspace"),
            weights_vec: DVector::zeros(IMRT_BEAM_COUNT),
            pareto_points: Vec::new(),
            show_pareto_curve: false,
            selected_idx: 0,
        }
    }
}

impl ImrtParetoController {
    /// Copies the `4 x 4096` unit-weight beam columns into `influence_matrix` without heap allocation.
    pub fn sync_influence_matrix(&mut self, ws: &ImrtWorkspace) {
        for b in 0..IMRT_BEAM_COUNT {
            let col_src = &ws.unit_beams[b];
            for (v, &val) in col_src.iter().enumerate() {
                self.influence_matrix[(v, b)] = val;
            }
        }
    }

    fn plan_config(prescription_gy: f64, oar_limit_gy: f64) -> ImrtPlanConfig {
        ImrtPlanConfig {
            prescription_gy,
            oar_limit_gy,
            alpha: IMRT_COST_ALPHA,
            beta_oar: IMRT_COST_BETA_OAR,
            max_beam_weight: IMRT_MAX_BEAM_WEIGHT,
        }
    }

    /// Runs preallocated `L-BFGS-B` on `[0, 6]^4` to optimize the 4 coplanar beam weights in-place.
    pub fn solve_constrained_weights(
        &mut self,
        ws: &ImrtWorkspace,
        beam_weights: &mut [f64; IMRT_BEAM_COUNT],
        prescription_gy: f64,
        oar_limit_gy: f64,
    ) -> Option<f64> {
        self.sync_influence_matrix(ws);
        for (b, &w) in beam_weights.iter().enumerate() {
            self.weights_vec[b] = w.clamp(0.0, IMRT_MAX_BEAM_WEIGHT);
        }
        let cfg = Self::plan_config(prescription_gy, oar_limit_gy);
        let cost = solve_imrt_weights_in_place(
            &self.influence_matrix,
            &ws.tumor_indices,
            &ws.organ_indices,
            &cfg,
            &mut self.weights_vec,
            &mut self.solver_ws,
        )
        .ok()?;
        for (b, dst) in beam_weights.iter_mut().enumerate() {
            *dst = self.weights_vec[b].clamp(0.0, IMRT_MAX_BEAM_WEIGHT);
        }
        Some(cost)
    }

    /// Computes the non-dominated Tumor-vs-OAR Pareto trade-off frontier across scalarized weights.
    pub fn recompute_pareto_curve(
        &mut self,
        ws: &ImrtWorkspace,
        prescription_gy: f64,
        oar_limit_gy: f64,
    ) {
        self.sync_influence_matrix(ws);
        let cfg = Self::plan_config(prescription_gy, oar_limit_gy);
        if let Ok(pts) = compute_imrt_pareto_frontier(
            &self.influence_matrix,
            &ws.tumor_indices,
            &ws.organ_indices,
            &cfg,
            IMRT_PARETO_SAMPLES,
        ) {
            self.pareto_points = pts;
            if self.selected_idx >= self.pareto_points.len() {
                self.selected_idx = self.pareto_points.len() / 2;
            }
        }
    }

    /// Applies a selected Pareto trade-off operating point onto `beam_weights`.
    pub fn apply_pareto_point(
        &self,
        idx: usize,
        beam_weights: &mut [f64; IMRT_BEAM_COUNT],
    ) -> bool {
        let Some(pt) = self.pareto_points.get(idx) else {
            return false;
        };
        for (b, dst) in beam_weights.iter_mut().enumerate() {
            if b < pt.weights.len() {
                *dst = pt.weights[b].clamp(0.0, IMRT_MAX_BEAM_WEIGHT);
            }
        }
        true
    }

    /// Renders side-panel controls for L-BFGS-B auto-solving and Pareto frontier navigation.
    pub fn draw_controls(
        &mut self,
        ui: &mut egui::Ui,
        ws: &ImrtWorkspace,
        beam_weights: &mut [f64; IMRT_BEAM_COUNT],
        prescription_gy: f64,
        oar_limit_gy: f64,
    ) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.strong("Constrained IMRT & Pareto Trade-off");
            let solve_tip = "Solve box-constrained 4-beam IMRT objective via L-BFGS-B";
            if ui
                .button("🎯 Auto-Solve Weights (L-BFGS-B)")
                .accessible_hover_text(solve_tip)
                .clicked()
            {
                if let Some(j_opt) =
                    self.solve_constrained_weights(ws, beam_weights, prescription_gy, oar_limit_gy)
                {
                    announce_live_status(
                        ui.ctx(),
                        format!("Solved constrained IMRT plan: J* = {j_opt:.1}"),
                    );
                    changed = true;
                }
            }
            let toggled = ui
                .checkbox(&mut self.show_pareto_curve, "Show Pareto Trade-off Curve")
                .changed();
            let recompute = ui.button("📈 Recompute Pareto Frontier").clicked();
            if recompute || (toggled && self.show_pareto_curve && self.pareto_points.is_empty()) {
                self.recompute_pareto_curve(ws, prescription_gy, oar_limit_gy);
            }
            changed |= self.draw_pareto_selector(ui, beam_weights);
        });
        changed
    }

    fn draw_pareto_selector(
        &mut self,
        ui: &mut egui::Ui,
        beam_weights: &mut [f64; IMRT_BEAM_COUNT],
    ) -> bool {
        if self.pareto_points.is_empty() {
            return false;
        }
        let max_idx = self.pareto_points.len().saturating_sub(1);
        let mut changed = false;
        if max_idx > 0 {
            let sl = egui::Slider::new(&mut self.selected_idx, 0..=max_idx).text("Pareto Plan #");
            if ui.add(sl).changed() {
                changed |= self.apply_pareto_point(self.selected_idx, beam_weights);
            }
        }
        if let Some(pt) = self.pareto_points.get(self.selected_idx) {
            ui.small(format!(
                "PTV = {:.1} Gy | OAR max = {:.1} Gy (β_OAR = {:.2})",
                pt.tumor_mean_gy, pt.oar_max_gy, pt.beta_oar
            ));
            if ui.small_button("Apply Selected Pareto Plan").clicked() {
                changed |= self.apply_pareto_point(self.selected_idx, beam_weights);
            }
        }
        changed
    }

    /// Renders the phantom heatmap and (when enabled) the Tumor-vs-OAR Pareto trade-off curve.
    pub fn draw_central_panels(
        &self,
        ui: &mut egui::Ui,
        imrt_mode: bool,
        texture: Option<egui::TextureHandle>,
        callouts: &mut CalloutManager,
    ) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        if imrt_mode && self.show_pareto_curve && !self.pareto_points.is_empty() {
            egui::TopBottomPanel::bottom(ui.id().with("imrt_pareto_bottom"))
                .resizable(true)
                .default_height(195.0)
                .min_height(130.0)
                .show_inside(ui, |p_ui| {
                    self.draw_pareto_plot(p_ui, &palette);
                });
        }
        egui::CentralPanel::default().show_inside(ui, |c_ui| {
            if let Some(tex) = texture {
                Plot::new("dose_heatmap")
                    .data_aspect(1.0)
                    .view_aspect(1.0)
                    .show_grid(false)
                    .show_axes([true, true])
                    .show(c_ui, |plot_ui| {
                        plot_ui.image(PlotImage::new(
                            "dose_field",
                            tex.id(),
                            PlotPoint::new(0.0, 0.0),
                            [IMRT_PHANTOM_SIZE_CM as f32, IMRT_PHANTOM_SIZE_CM as f32],
                        ));
                        draw_imrt_contours(imrt_mode, &palette, plot_ui);
                        callouts.render_on_plot(plot_ui);
                    });
            }
        });
    }

    fn draw_pareto_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let mut sorted: Vec<[f64; 2]> = self
            .pareto_points
            .iter()
            .map(|p| [p.oar_overdose_cost, p.tumor_quadratic_cost])
            .collect();
        sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));

        Plot::new("imrt_pareto_frontier_plot")
            .x_axis_label("OAR Overdose Penalty")
            .y_axis_label("PTV Target Squared Error")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Pareto Frontier", sorted.clone())
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                plot_ui.points(
                    Points::new("Non-Dominated Plans", sorted)
                        .color(palette.series_color(0))
                        .radius(4.5_f32),
                );
                if let Some(sel) = self.pareto_points.get(self.selected_idx) {
                    plot_ui.points(
                        Points::new(
                            "Selected Plan",
                            vec![[sel.oar_overdose_cost, sel.tumor_quadratic_cost]],
                        )
                        .color(palette.success)
                        .radius(7.0_f32)
                        .shape(MarkerShape::Diamond),
                    );
                }
            });
    }
}

fn draw_imrt_contours(imrt_mode: bool, palette: &SemanticPalette, plot_ui: &mut PlotUi<'_>) {
    if !imrt_mode {
        return;
    }
    let ring = |c: [f64; 2], r: f64| -> Vec<[f64; 2]> {
        (0..32)
            .map(|k| {
                let a = std::f64::consts::TAU * (k as f64) / 32.0;
                [c[0] + r * a.cos(), c[1] + r * a.sin()]
            })
            .collect()
    };
    let ptv = Polygon::new(
        "PTV Target (D_Rx)",
        ring(IMRT_TUMOR_CENTER, IMRT_TUMOR_RADIUS_CM),
    )
    .fill_color(Color32::TRANSPARENT)
    .stroke(egui::Stroke::new(2.0_f32, palette.success));
    let oar = Polygon::new("OAR (D_lim)", ring(IMRT_OAR_CENTER, IMRT_OAR_RADIUS_CM))
        .fill_color(Color32::TRANSPARENT)
        .stroke(egui::Stroke::new(2.0_f32, palette.error));
    plot_ui.polygon(ptv);
    plot_ui.polygon(oar);
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::super::dose::DoseCalculationTool;

    #[test]
    fn test_dose_tool_constrained_imrt_solve_and_pareto_curve() {
        let mut tool = DoseCalculationTool::default();
        tool.imrt_mode = true;
        tool.refresh_model();
        let initial_cost = tool.imrt_workspace.summary.cost_j;

        let solved_j = tool
            .pareto_controller
            .solve_constrained_weights(
                &tool.imrt_workspace,
                &mut tool.beam_weights,
                tool.prescription_gy,
                tool.oar_limit_gy,
            )
            .expect("constrained IMRT solve");
        tool.refresh_model();
        assert!(solved_j < initial_cost * 0.5);
        assert!(tool
            .beam_weights
            .iter()
            .all(|&w| (0.0..=super::IMRT_MAX_BEAM_WEIGHT).contains(&w)));

        tool.pareto_controller.recompute_pareto_curve(
            &tool.imrt_workspace,
            tool.prescription_gy,
            tool.oar_limit_gy,
        );
        assert!(tool.pareto_controller.pareto_points.len() >= 3);
        assert!(tool
            .pareto_controller
            .apply_pareto_point(0, &mut tool.beam_weights));
    }
}

// [cite:lorahub]
