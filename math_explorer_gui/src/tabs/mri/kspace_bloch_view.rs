#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_tool::MriKSpaceTool;
use super::kspace_types::ISOCHROMAT_COUNT;
use crate::accessibility::AccessibleTheoryHover;
use crate::framework::Plot3DViewport;
use eframe::egui;
use egui_plot::{Line, PlotPoint, Points};

impl MriKSpaceTool {
    pub(crate) fn draw_selected_voxel_telemetry(&self, ui: &mut egui::Ui) {
        let [sr, sc] = self.config.selected_voxel;
        let [rho, t1, t2, db0] = self.latest.selected_tissue;
        ui.label(format!(
            "t = {:.3} s | Voxel ({sr}, {sc})",
            self.latest.current_time
        ));
        if let Some(tau) = self.latest.echo_tau {
            ui.label(format!("Spin-Echo τ = {tau:.3}s (TE = {:.3}s)", 2.0 * tau));
        }
        ui.small(format!(
            "ρ={rho:.2}, T1={t1:.2}s, T2={t2:.3}s, ΔB0={db0:.2e}T"
        ));
    }

    pub(crate) fn paint_voxel_crosshair(&self, ui: &egui::Ui, rect: egui::Rect) {
        let n = self.config.grid_size.max(1) as f32;
        let [sr, sc] = self.config.selected_voxel;
        let x = rect.min.x + ((sc as f32) + 0.5) / n * rect.width();
        let y = rect.min.y + ((sr as f32) + 0.5) / n * rect.height();
        let accent = crate::app::SemanticPalette::from_visuals(ui.visuals()).accent;
        ui.painter().circle_stroke(
            egui::pos2(x, y),
            4.5_f32,
            egui::Stroke::new(1.5_f32, accent),
        );
    }

    pub(crate) fn draw_bloch_3d_readout(&mut self, ui: &mut egui::Ui) {
        ui.heading("Selected Voxel 3D Bloch Sphere Readout");
        let equator = project_sphere_ring(&self.config.camera, 0);
        let meridian_a = project_sphere_ring(&self.config.camera, 1);
        let meridian_b = project_sphere_ring(&self.config.camera, 2);
        let proj = |p: [f64; 3]| {
            let [x, y] = self.config.camera.project(&p);
            PlotPoint::new(x, y)
        };
        let origin = proj([0.0, 0.0, 0.0]);
        let net = self.latest.selected_net_m;
        let net_end = proj([net.x, net.y, net.z]);
        let net_line = [origin, net_end];
        let net_tip = [net_end];
        let x_ax = [origin, proj([1.15, 0.0, 0.0])];
        let y_ax = [origin, proj([0.0, 1.15, 0.0])];
        let z_ax = [origin, proj([0.0, 0.0, 1.15])];
        let iso_lines: [[PlotPoint; 2]; ISOCHROMAT_COUNT] = std::array::from_fn(|i| {
            let s = self.latest.selected_spins[i];
            [origin, proj([s.x, s.y, s.z])]
        });
        let trail: Vec<PlotPoint> = self
            .latest
            .selected_trajectory
            .iter()
            .map(|m| proj([m.x, m.y, m.z]))
            .collect();

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let resp = Plot3DViewport::new("mri_voxel_bloch_3d", 1.4)
            .show(ui, &mut self.config.camera, |plot_ui| {
                plot_ui.line(Line::new("", &equator[..]).color(palette.graph_edge));
                plot_ui.line(Line::new("", &meridian_a[..]).color(palette.graph_edge));
                plot_ui.line(Line::new("", &meridian_b[..]).color(palette.graph_edge));
                plot_ui.line(Line::new("Mx", &x_ax[..]).color(palette.axis_x));
                plot_ui.line(Line::new("My", &y_ax[..]).color(palette.axis_y));
                plot_ui.line(Line::new("Mz", &z_ax[..]).color(palette.axis_z));
                for line in &iso_lines {
                    plot_ui.line(
                        Line::new("", &line[..])
                            .color(palette.warning)
                            .width(1.5_f32),
                    );
                }
                if trail.len() > 1 {
                    plot_ui.line(Line::new("Trajectory", &trail[..]).color(palette.axis_y));
                }
                plot_ui.line(
                    Line::new("Net M", &net_line[..])
                        .color(palette.accent)
                        .width(3.0_f32),
                );
                plot_ui.points(
                    Points::new("", &net_tip[..])
                        .radius(5.0_f32)
                        .color(palette.accent),
                );
            })
            .response;
        resp.accessible_theory_hover(&self.theory);
    }
}

fn project_sphere_ring(camera: &crate::framework::Camera3D, axis: usize) -> [PlotPoint; 61] {
    let mut pts = [PlotPoint::new(0.0, 0.0); 61];
    for (i, slot) in pts.iter_mut().enumerate() {
        let theta = (i as f64) * std::f64::consts::TAU / 60.0;
        let p3 = match axis {
            0 => [theta.cos(), theta.sin(), 0.0],
            1 => [0.0, theta.cos(), theta.sin()],
            _ => [theta.cos(), 0.0, theta.sin()],
        };
        let [x, y] = camera.project(&p3);
        *slot = PlotPoint::new(x, y);
    }
    pts
}
