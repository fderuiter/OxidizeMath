//! Decision-boundary probability raster, `MarchingSquares2D` `p = 0.50` iso-contour,
//! and live training plots for [`super::TrainingMonitorTool`].

use std::sync::Arc;

use crate::accessibility::register_canvas_accessibility;
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui::{self, Color32, Stroke};
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotImage, PlotPoint, Points};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::ai::deep_learning_theory::cycle::TrainingLoop;
use math_explorer::ai::deep_learning_theory::linear_algebra::Vector;
use math_explorer::ai::deep_learning_theory::model::TwoLayerMLP;
use math_explorer::pure_math::analysis::pde::wave_2d::MarchingSquares2D;

use super::TrainingMonitorTool;

pub(crate) const BOUNDARY_X_MIN: f64 = -2.25;
pub(crate) const BOUNDARY_X_MAX: f64 = 2.25;
pub(crate) const BOUNDARY_Y_MIN: f64 = -1.35;
pub(crate) const BOUNDARY_Y_MAX: f64 = 1.35;
const GRID_COLS: usize = 84;
const GRID_ROWS: usize = 52;
const RASTER_W: usize = 168;
const RASTER_H: usize = 104;

pub(crate) struct DecisionBoundaryColors {
    pub(crate) class_0_bg: Color32,
    pub(crate) class_1_bg: Color32,
    pub(crate) contour: Color32,
}

pub(crate) const fn boundary_colors_for_theme(theme: ThemeMode) -> DecisionBoundaryColors {
    match theme {
        ThemeMode::Light => DecisionBoundaryColors {
            class_0_bg: Color32::from_rgb(254, 232, 232),
            class_1_bg: Color32::from_rgb(226, 238, 254),
            contour: Color32::from_rgb(24, 34, 52),
        },
        ThemeMode::Dark => DecisionBoundaryColors {
            class_0_bg: Color32::from_rgb(54, 22, 28),
            class_1_bg: Color32::from_rgb(20, 34, 58),
            contour: Color32::from_rgb(241, 245, 249),
        },
        ThemeMode::HighContrast => DecisionBoundaryColors {
            class_0_bg: Color32::from_rgb(36, 8, 12),
            class_1_bg: Color32::from_rgb(6, 18, 38),
            contour: Color32::WHITE,
        },
    }
}

pub(crate) fn lerp_class_bg(colors: &DecisionBoundaryColors, p_class_0: f64) -> Color32 {
    let t = p_class_0.clamp(0.0, 1.0);
    let r = (colors.class_1_bg.r() as f64 * (1.0 - t) + colors.class_0_bg.r() as f64 * t)
        .round()
        .clamp(0.0, 255.0) as u8;
    let g = (colors.class_1_bg.g() as f64 * (1.0 - t) + colors.class_0_bg.g() as f64 * t)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (colors.class_1_bg.b() as f64 * (1.0 - t) + colors.class_0_bg.b() as f64 * t)
        .round()
        .clamp(0.0, 255.0) as u8;
    Color32::from_rgb(r, g, b)
}

pub(crate) struct DecisionBoundaryCache {
    cached_revision: Option<u64>,
    pub(crate) prob_class_0: Vec<f64>,
    signed_field: Vec<f64>,
    wall_mask: Vec<f64>,
    marching_squares: MarchingSquares2D,
    pub(crate) contour_segments: Arc<Vec<Vec<PlotPoint>>>,
    pub(crate) contour_bounds: PlotBounds,
    raster_key: Option<(u64, ThemeMode)>,
    raster_image: Option<Arc<egui::ColorImage>>,
    raster_texture: Option<egui::TextureHandle>,
}

impl Default for DecisionBoundaryCache {
    fn default() -> Self {
        let cells = GRID_COLS * GRID_ROWS;
        Self {
            cached_revision: None,
            prob_class_0: vec![0.5; cells],
            signed_field: vec![0.0; cells],
            wall_mask: vec![1.0; cells],
            marching_squares: MarchingSquares2D::new(cells * 2),
            contour_segments: Arc::new(Vec::new()),
            contour_bounds: PlotBounds::NOTHING,
            raster_key: None,
            raster_image: None,
            raster_texture: None,
        }
    }
}

impl DecisionBoundaryCache {
    pub(crate) fn invalidate(&mut self) {
        self.cached_revision = None;
        self.raster_key = None;
    }

    pub(crate) fn ensure_evaluated(
        &mut self,
        revision: u64,
        training_loop: &TrainingLoop<TwoLayerMLP>,
    ) {
        if self.cached_revision == Some(revision) {
            return;
        }
        let span_x = BOUNDARY_X_MAX - BOUNDARY_X_MIN;
        let span_y = BOUNDARY_Y_MAX - BOUNDARY_Y_MIN;
        let denom_x = (GRID_COLS - 1) as f64;
        let denom_y = (GRID_ROWS - 1) as f64;
        let mut probe = Vector::from_vec(vec![0.0, 0.0]);
        for gy in 0..GRID_ROWS {
            let wy = BOUNDARY_Y_MIN + (gy as f64 / denom_y) * span_y;
            for gx in 0..GRID_COLS {
                let wx = BOUNDARY_X_MIN + (gx as f64 / denom_x) * span_x;
                probe[0] = wx;
                probe[1] = wy;
                let pred = training_loop.predict(&probe);
                let p0 = pred.get(0).copied().unwrap_or(0.5).clamp(0.0, 1.0);
                let idx = flatten_2d_index(gx, gy, GRID_COLS);
                self.prob_class_0[idx] = p0;
                self.signed_field[idx] = p0 - 0.5;
            }
        }
        let raw = self.marching_squares.extract_zero_contours(
            &self.signed_field,
            &self.wall_mask,
            GRID_COLS,
            GRID_ROWS,
            1.0e-12,
        );
        let mut bounds = PlotBounds::NOTHING;
        let mut segs = Vec::with_capacity(raw.len());
        for &[p0, p1] in raw {
            let a = PlotPoint::new(
                BOUNDARY_X_MIN + (p0[0] / denom_x) * span_x,
                BOUNDARY_Y_MIN + (p0[1] / denom_y) * span_y,
            );
            let b = PlotPoint::new(
                BOUNDARY_X_MIN + (p1[0] / denom_x) * span_x,
                BOUNDARY_Y_MIN + (p1[1] / denom_y) * span_y,
            );
            bounds.extend_with(&a);
            bounds.extend_with(&b);
            segs.push(vec![a, b]);
        }
        self.contour_segments = Arc::new(segs);
        self.contour_bounds = bounds;
        self.cached_revision = Some(revision);
        self.raster_key = None;
    }

    fn bilinear_prob(&self, u: f64, v: f64) -> f64 {
        let fx = (u.clamp(0.0, 1.0) * (GRID_COLS - 1) as f64).max(0.0);
        let fy = (v.clamp(0.0, 1.0) * (GRID_ROWS - 1) as f64).max(0.0);
        let c0 = (fx.floor() as usize).min(GRID_COLS - 1);
        let r0 = (fy.floor() as usize).min(GRID_ROWS - 1);
        let c1 = (c0 + 1).min(GRID_COLS - 1);
        let r1 = (r0 + 1).min(GRID_ROWS - 1);
        let tx = fx - c0 as f64;
        let ty = fy - r0 as f64;
        let bot = self.prob_class_0[flatten_2d_index(c0, r0, GRID_COLS)] * (1.0 - tx)
            + self.prob_class_0[flatten_2d_index(c1, r0, GRID_COLS)] * tx;
        let top = self.prob_class_0[flatten_2d_index(c0, r1, GRID_COLS)] * (1.0 - tx)
            + self.prob_class_0[flatten_2d_index(c1, r1, GRID_COLS)] * tx;
        bot * (1.0 - ty) + top * ty
    }

    fn ensure_raster(
        &mut self,
        ctx: &egui::Context,
        revision: u64,
        theme: ThemeMode,
    ) -> (egui::TextureId, Arc<egui::ColorImage>) {
        if self.raster_key != Some((revision, theme)) || self.raster_texture.is_none() {
            let colors = boundary_colors_for_theme(theme);
            let denom_x = (RASTER_W - 1) as f64;
            let denom_y = (RASTER_H - 1) as f64;
            let mut pixels = Vec::with_capacity(RASTER_W * RASTER_H);
            for py in 0..RASTER_H {
                let v = 1.0 - (py as f64 / denom_y);
                for px in 0..RASTER_W {
                    let u = px as f64 / denom_x;
                    pixels.push(lerp_class_bg(&colors, self.bilinear_prob(u, v)));
                }
            }
            let image = Arc::new(egui::ColorImage::new([RASTER_W, RASTER_H], pixels));
            let handle = ctx.load_texture(
                "training_monitor_decision_surface",
                (*image).clone(),
                egui::TextureOptions::LINEAR,
            );
            self.raster_image = Some(image);
            self.raster_texture = Some(handle);
            self.raster_key = Some((revision, theme));
        }
        (
            self.raster_texture.as_ref().expect("texture").id(),
            Arc::clone(self.raster_image.as_ref().expect("image")),
        )
    }
}

impl TrainingMonitorTool {
    pub(crate) fn ensure_boundary_cache(&mut self) {
        self.boundary_cache
            .ensure_evaluated(self.model_revision, &self.training_loop);
    }

    pub(super) fn draw_plots(&mut self, parent_ui: &mut egui::Ui) {
        self.ensure_boundary_cache();
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.draw_history_row(ui);
            ui.separator();
            self.draw_decision_boundary_plot(ui);
        });
    }

    fn draw_history_row(&self, ui: &mut egui::Ui) {
        ui.columns(2, |columns| {
            let loss_palette = SemanticPalette::from_visuals(columns[0].visuals());
            columns[0].heading("Loss History");
            Plot::new("loss_plot")
                .height(185.0)
                .include_y(0.0)
                .show(&mut columns[0], |plot_ui| {
                    plot_ui.line(
                        Line::new("Loss", self.loss_history.clone())
                            .color(loss_palette.info)
                            .width(2.0_f32)
                            .fill(0.0_f32),
                    );
                    if let Some(&last) = self.loss_history.last() {
                        plot_ui.points(
                            Points::new("Current Loss", vec![last])
                                .color(loss_palette.info)
                                .radius(4.0_f32)
                                .companion_visual_only(true),
                        );
                    }
                });

            let acc_palette = SemanticPalette::from_visuals(columns[1].visuals());
            columns[1].heading("Accuracy History");
            Plot::new("accuracy_plot")
                .height(185.0)
                .include_y(0.0)
                .include_y(1.0)
                .show(&mut columns[1], |plot_ui| {
                    plot_ui.line(
                        Line::new("Accuracy", self.accuracy_history.clone())
                            .color(acc_palette.success)
                            .width(2.0_f32)
                            .fill(0.0_f32),
                    );
                    if let Some(&last) = self.accuracy_history.last() {
                        plot_ui.points(
                            Points::new("Current Accuracy", vec![last])
                                .color(acc_palette.success)
                                .radius(4.0_f32)
                                .companion_visual_only(true),
                        );
                    }
                });
        });
    }

    fn draw_decision_boundary_plot(&mut self, ui: &mut egui::Ui) {
        let theme = ThemeMode::from_visuals(ui.visuals());
        let palette = SemanticPalette::for_theme(theme);
        let b_colors = boundary_colors_for_theme(theme);
        let raster = self
            .boundary_cache
            .ensure_raster(ui.ctx(), self.model_revision, theme);

        ui.heading("Decision Boundary Visualization");
        let resp = Plot::new("data_plot")
            .data_aspect(1.0)
            .include_x(-1.25)
            .include_x(1.25)
            .include_y(-1.25)
            .include_y(1.25)
            .legend(Legend::default().position(Corner::LeftTop))
            .show(ui, |plot_ui| {
                self.render_boundary_layers(plot_ui, &palette, b_colors.contour, raster);
            });

        let summary = format!(
            "Epoch {}, {} contour segments on p = 0.50 boundary",
            self.epoch,
            self.boundary_cache.contour_segments.len()
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            egui::accesskit::Role::GraphicsDocument,
            "MLP spiral decision boundary canvas",
            Some(&summary),
        );
    }

    fn render_boundary_layers(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
        contour_color: Color32,
        raster: (egui::TextureId, Arc<egui::ColorImage>),
    ) {
        let span_x = (BOUNDARY_X_MAX - BOUNDARY_X_MIN) as f32;
        let span_y = (BOUNDARY_Y_MAX - BOUNDARY_Y_MIN) as f32;
        plot_ui.image(
            PlotImage::new("", raster.0, PlotPoint::new(0.0, 0.0), [span_x, span_y])
                .allow_hover(false)
                .companion_visual_only(true)
                .with_export_image(
                    raster.1,
                    [
                        BOUNDARY_X_MIN,
                        BOUNDARY_X_MAX,
                        BOUNDARY_Y_MIN,
                        BOUNDARY_Y_MAX,
                    ],
                ),
        );
        let contour_segs = Arc::clone(&self.boundary_cache.contour_segments);
        if !contour_segs.is_empty() {
            plot_ui.add(
                Line::from_segments("p = 0.50 Boundary", contour_segs)
                    .with_bounds(self.boundary_cache.contour_bounds)
                    .stroke(Stroke::new(2.2_f32, contour_color))
                    .allow_hover(false)
                    .companion_visual_only(true),
            );
        } else {
            plot_ui.line(
                Line::new("p = 0.50 Boundary", Vec::<[f64; 2]>::new())
                    .color(contour_color)
                    .width(2.2_f32),
            );
        }
        let (c0, c1) = self.split_spiral_classes();
        plot_ui.points(
            Points::new("Class 0", c0)
                .color(palette.axis_x)
                .radius(4.0_f32),
        );
        plot_ui.points(
            Points::new("Class 1", c1)
                .color(palette.axis_z)
                .radius(4.0_f32),
        );
    }

    fn split_spiral_classes(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let mut class_0 = Vec::with_capacity(self.data.len() / 2);
        let mut class_1 = Vec::with_capacity(self.data.len() / 2);
        for (x, y) in &self.data {
            let pt = [x[0], x[1]];
            if y[0] > 0.5 {
                class_0.push(pt);
            } else {
                class_1.push(pt);
            }
        }
        (class_0, class_1)
    }
}

// [cite:graph_parameters_rust]
