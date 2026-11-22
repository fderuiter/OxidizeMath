#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! UI tables and radial/polar profile plots for [`ChristoffelInspectorTool`].

use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::{PaneSidePanel, scoped_panel_id};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use math_explorer::nalgebra::SVector;
use math_explorer::physics::high_energy::{
    KerrEquatorialMetric, MorrisThorneWormhole, SchwarzschildBlackHole,
};
use math_explorer::pure_math::tensor::{StaticMetric, TwoSphereMetric};

use super::{ChristoffelEvaluation, ChristoffelInspectorTool, InspectorManifold};

impl ChristoffelInspectorTool {
    pub(super) fn render_inspector(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_inspector_side_panel(parent_ui);
        let eval = self.evaluate_current();
        self.draw_inspector_central_panel(parent_ui, eval.as_ref());
    }

    fn draw_inspector_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "relativity_inspector_side");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Christoffel Inspector");
            ui.separator();

            let mut changed = self.draw_manifold_controls(ui);
            ui.separator();
            ui.label("Upper Index mu in Gamma^mu_(alpha beta):");
            let labels = self.state.manifold.coord_labels();
            ui.horizontal_wrapped(|ui| {
                for (mu, &coord) in labels.iter().enumerate() {
                    if ui
                        .selectable_value(
                            &mut self.state.selected_mu,
                            mu,
                            format!("mu={mu} ({coord})"),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                }
            });

            if changed {
                self.clamp_in_place();
                self.revision = self.revision.wrapping_add(1);
            }
        });
    }

    fn draw_manifold_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_label("Manifold")
            .selected_text(self.state.manifold.label())
            .show_ui(ui, |ui| {
                for m in [
                    InspectorManifold::Schwarzschild4D,
                    InspectorManifold::KerrEquatorial4D,
                    InspectorManifold::MorrisThorne3D,
                    InspectorManifold::TwoSphere2D,
                ] {
                    if ui
                        .selectable_value(&mut self.state.manifold, m, m.label())
                        .changed()
                    {
                        changed = true;
                    }
                }
            });

        changed |= ui
            .add(egui::Slider::new(&mut self.state.scale_radius, 0.5..=5.0).text("Scale R"))
            .changed();
        if self.state.manifold == InspectorManifold::KerrEquatorial4D {
            changed |= ui
                .add(egui::Slider::new(&mut self.state.spin_chi, -0.92..=0.92).text("Spin chi"))
                .changed();
        }
        if self.state.manifold != InspectorManifold::TwoSphere2D {
            let r_range = match self.state.manifold {
                InspectorManifold::MorrisThorne3D => -10.0..=10.0,
                _ => (1.15 * self.state.scale_radius)..=16.0,
            };
            changed |= ui
                .add(egui::Slider::new(&mut self.state.radial_coord, r_range).text("Radial r/l"))
                .changed();
        }
        if self.state.manifold != InspectorManifold::MorrisThorne3D {
            changed |= ui
                .add(egui::Slider::new(&mut self.state.theta, 0.15..=2.99).text("Angle theta"))
                .changed();
        }
        changed
    }

    fn draw_inspector_central_panel(
        &self,
        parent_ui: &mut egui::Ui,
        eval: Option<&ChristoffelEvaluation>,
    ) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            if let Some(ev) = eval {
                self.draw_matrix_comparison_grid(ui, ev, &palette);
            } else {
                ui.colored_label(palette.warning, "Point lies inside horizon or singular locus.");
            }
            ui.separator();
            ui.strong("Connection Component Profiles Along Coordinate Curve");
            let datasets = self.profile_datasets();
            let plot_id = scoped_panel_id(ui, "christoffel_profile_plot");
            let response = Plot::new(plot_id).allow_boxed_zoom(false).show(ui, |plot_ui| {
                for (idx, (name, pts)) in datasets.into_iter().enumerate() {
                    plot_ui.line(
                        Line::new(name, PlotPoints::new(pts))
                            .color(palette.series_color(idx))
                            .width(2.0_f32),
                    );
                }
                let x_cur = if self.state.manifold == InspectorManifold::TwoSphere2D {
                    self.state.theta
                } else {
                    self.state.radial_coord
                };
                plot_ui.points(
                    Points::new("Evaluation Coordinate", vec![[x_cur, 0.0]])
                        .radius(4.5_f32)
                        .color(palette.accent),
                );
            });
            response.response.accessible_theory_hover(self);
        });
    }

    fn draw_matrix_comparison_grid(
        &self,
        ui: &mut egui::Ui,
        ev: &ChristoffelEvaluation,
        palette: &SemanticPalette,
    ) {
        let coords = self.state.manifold.coord_labels();
        let mu = self.state.selected_mu.min(ev.dim - 1);
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("Slice Gamma^{}_(alpha beta) (Analytical)", coords[mu]));
            ui.separator();
            ui.colored_label(
                palette.success,
                format!(
                    "Max |Analytical - CentralDiff| across all indices: {:.3e}",
                    ev.max_abs_diff
                ),
            );
        });

        let grid_id = scoped_panel_id(ui, "christoffel_matrix_grid");
        egui::Grid::new(grid_id).striped(true).spacing([14.0, 6.0]).show(ui, |ui| {
            ui.strong("alpha \\ beta");
            for &col in coords {
                ui.strong(col);
            }
            ui.strong("Max Row Residual");
            ui.end_row();

            for (a, &row_label) in coords.iter().enumerate() {
                ui.strong(row_label);
                let mut row_err = 0.0_f64;
                for b in 0..ev.dim {
                    let val = ev.analytical[mu][a][b];
                    let diff = (val - ev.numerical[mu][a][b]).abs();
                    row_err = row_err.max(diff);
                    let color = if val.abs() > 1e-9 {
                        palette.info
                    } else {
                        palette.muted_text
                    };
                    ui.colored_label(color, format!("{val:+.6}"));
                }
                ui.colored_label(palette.muted_text, format!("{row_err:.2e}"));
                ui.end_row();
            }
        });
    }

    pub(super) fn profile_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let n = 64usize;
        let rs = self.state.scale_radius;
        match self.state.manifold {
            InspectorManifold::Schwarzschild4D => self.schwarzschild_profiles(rs, n),
            InspectorManifold::KerrEquatorial4D => self.kerr_profiles(rs, n),
            InspectorManifold::MorrisThorne3D => Self::wormhole_profiles(rs, n),
            InspectorManifold::TwoSphere2D => Self::sphere_profiles(rs, n),
        }
    }

    fn schwarzschild_profiles(&self, rs: f64, n: usize) -> Vec<(String, Vec<[f64; 2]>)> {
        let Ok(bh) = SchwarzschildBlackHole::from_schwarzschild_radius(rs) else {
            return Vec::new();
        };
        let mut g00 = Vec::with_capacity(n);
        let mut g11 = Vec::with_capacity(n);
        for i in 0..n {
            let r = 1.15 * rs + 14.0 * (i as f64) / ((n - 1) as f64);
            let pt = SVector::<f64, 4>::new(0.0, r, self.state.theta, 0.0);
            if let Ok(g) = bh.christoffel_at(&pt) {
                g00.push([r, g[1][(0, 0)]]);
                g11.push([r, g[0][(0, 1)]]);
            }
        }
        vec![
            ("Gamma^r_(ct ct)(r)".to_string(), g00),
            ("Gamma^ct_(ct r)(r)".to_string(), g11),
        ]
    }

    fn kerr_profiles(&self, rs: f64, n: usize) -> Vec<(String, Vec<[f64; 2]>)> {
        let Ok(kerr) = KerrEquatorialMetric::new(rs, 0.5 * rs * self.state.spin_chi) else {
            return Vec::new();
        };
        let mut g_drag = Vec::with_capacity(n);
        let mut g_rad = Vec::with_capacity(n);
        for i in 0..n {
            let r = 1.15 * kerr.outer_horizon_radius() + 14.0 * (i as f64) / ((n - 1) as f64);
            let pt = SVector::<f64, 4>::new(0.0, r, self.state.theta, 0.0);
            if let Ok(g) = kerr.christoffel_at(&pt) {
                g_drag.push([r, g[3][(0, 1)]]);
                g_rad.push([r, g[1][(0, 0)]]);
            }
        }
        vec![
            ("Gamma^phi_(ct r)(r) Frame Drag".to_string(), g_drag),
            ("Gamma^r_(ct ct)(r)".to_string(), g_rad),
        ]
    }

    fn wormhole_profiles(rs: f64, n: usize) -> Vec<(String, Vec<[f64; 2]>)> {
        let Ok(wh) = MorrisThorneWormhole::new(rs) else {
            return Vec::new();
        };
        let mut g_phi = Vec::with_capacity(n);
        let mut g_l = Vec::with_capacity(n);
        for i in 0..n {
            let l = -8.0 + 16.0 * (i as f64) / ((n - 1) as f64);
            let pt = SVector::<f64, 3>::new(0.0, l, 0.0);
            if let Ok(g) = wh.christoffel_at(&pt) {
                g_phi.push([l, g[2][(1, 2)]]);
                g_l.push([l, 0.1 * g[1][(2, 2)]]);
            }
        }
        vec![
            ("Gamma^phi_(l phi)(l)".to_string(), g_phi),
            ("0.1 * Gamma^l_(phi phi)(l)".to_string(), g_l),
        ]
    }

    fn sphere_profiles(rs: f64, n: usize) -> Vec<(String, Vec<[f64; 2]>)> {
        let Ok(sp) = TwoSphereMetric::new(rs) else {
            return Vec::new();
        };
        let mut g_th = Vec::with_capacity(n);
        let mut g_ph = Vec::with_capacity(n);
        for i in 0..n {
            let th = 0.2 + 2.74 * (i as f64) / ((n - 1) as f64);
            let pt = SVector::<f64, 2>::new(th, 0.0);
            if let Ok(g) = sp.christoffel_at(&pt) {
                g_th.push([th, g[0][(1, 1)]]);
                g_ph.push([th, g[1][(0, 1)]]);
            }
        }
        vec![
            ("Gamma^theta_(phi phi)(theta)".to_string(), g_th),
            ("Gamma^phi_(theta phi)(theta)".to_string(), g_ph),
        ]
    }
}
