#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_tool::MriKSpaceTool;
use super::kspace_types::{MriKSpaceSnapshot, ISOCHROMAT_COUNT};
use crate::accessibility::AccessibleTheoryHover;
use crate::async_sim::SimWorkerState;
use crate::framework::{bounded_drag_value, Plot3DViewport};
use eframe::egui;
use egui_plot::{Line, PlotPoint, PlotPoints, Points};

impl MriKSpaceTool {
    pub(crate) fn draw_selected_voxel_telemetry(&mut self, ui: &mut egui::Ui) {
        let is_faulted = self.worker.state() == SimWorkerState::Faulted;
        if is_faulted {
            if !self.fault_notified {
                self.fault_notified = true;
                crate::audio::notify_solver_diverged();
            }
        } else {
            self.fault_notified = false;
        }
        let frame = self.latest_sonification;
        crate::audio::render_sonification_panel(ui, "mri_kspace", &frame, is_faulted);
        ui.separator();

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
        crate::reflective_ui::render_copyable_metric(
            ui,
            "Recon NRMSE vs ρ",
            &format!("{:.1}%", self.latest.nrmse_pct),
        );
        crate::reflective_ui::render_copyable_metric(
            ui,
            "Scan Acceleration R",
            &format!("{:.2}x", self.latest.acceleration_r),
        );
    }

    pub(crate) fn draw_gradient_controls(&mut self, ui: &mut egui::Ui) {
        let mut grad_changed = false;
        ui.horizontal(|ui| {
            ui.label("∫Gx dt");
            grad_changed |= ui
                .add(bounded_drag_value(
                    &mut self.config.grad_gx_integral,
                    -5.0e-5..=5.0e-5,
                    5.0e-7,
                ))
                .changed();
            ui.label("∫Gy dt");
            grad_changed |= ui
                .add(bounded_drag_value(
                    &mut self.config.grad_gy_integral,
                    -5.0e-5..=5.0e-5,
                    5.0e-7,
                ))
                .changed();
        });
        if grad_changed {
            self.sync_and_notify("sync_view");
        }
        let [kx, ky] = self.latest.k_coord_cycles_per_m;
        ui.small(format!(
            "k = ({kx:.1}, {ky:.1}) m⁻¹ | φ_grad = {:.2} rad",
            self.latest.selected_grad_phase_rad
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
        self.sync_kspace_callouts();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let cam_snap = self.config.camera;
        let sidecar = self.build_bloch_3d_sidecar(&cam_snap, &palette);
        let equator = project_sphere_ring(&cam_snap, 0);
        let meridian_a = project_sphere_ring(&cam_snap, 1);
        let meridian_b = project_sphere_ring(&cam_snap, 2);

        let latest = &self.latest;
        let callout_manager = &mut self.callout_manager;
        let resp = Plot3DViewport::new("mri_voxel_bloch_3d", 1.4)
            .show(ui, &mut self.config.camera, |plot_ui| {
                plot_ui.line(Line::new("", equator).color(palette.graph_edge));
                plot_ui.line(Line::new("", meridian_a).color(palette.graph_edge));
                plot_ui.line(Line::new("", meridian_b).color(palette.graph_edge));
                draw_bloch_vectors_3d(latest, plot_ui, &cam_snap, &palette);
                callout_manager.render_on_plot(plot_ui);
                plot_ui.set_3d_sidecar(sidecar);
            })
            .response;
        resp.accessible_theory_hover(&self.theory);
    }

    fn build_bloch_3d_sidecar(
        &self,
        camera: &crate::framework::Camera3D,
        palette: &crate::app::SemanticPalette,
    ) -> egui_plot::Plot3DSeriesSidecar {
        let mut sidecar = egui_plot::Plot3DSeriesSidecar::new();
        let origin_2d = camera.project(&[0.0, 0.0, 0.0]);
        let net = self.latest.selected_net_m;
        let net_w = [net.x, net.y, net.z];
        sidecar.push_series(egui_plot::Plot3DSeries::new(
            "Net M",
            vec![origin_2d, camera.project(&net_w)],
            vec![[0.0, 0.0, 0.0], net_w],
            palette.accent,
        ));
        let iso_world: Vec<[f64; 3]> = self
            .latest
            .selected_spins
            .iter()
            .map(|s| [s.x, s.y, s.z])
            .collect();
        let iso_proj: Vec<[f64; 2]> = iso_world.iter().map(|p| camera.project(p)).collect();
        sidecar.push_series(egui_plot::Plot3DSeries::new(
            "Isochromats",
            iso_proj,
            iso_world,
            palette.warning,
        ));
        if self.latest.selected_trajectory.len() > 1 {
            let t_world: Vec<[f64; 3]> = self
                .latest
                .selected_trajectory
                .iter()
                .map(|m| [m.x, m.y, m.z])
                .collect();
            let t_proj: Vec<[f64; 2]> = t_world.iter().map(|p| camera.project(p)).collect();
            sidecar.push_series(egui_plot::Plot3DSeries::new(
                "Trajectory",
                t_proj,
                t_world,
                palette.axis_y,
            ));
        }
        for (name, tip, color) in [
            ("Mx", [1.15, 0.0, 0.0], palette.axis_x),
            ("My", [0.0, 1.15, 0.0], palette.axis_y),
            ("Mz", [0.0, 0.0, 1.15], palette.axis_z),
        ] {
            sidecar.push_series(egui_plot::Plot3DSeries::new(
                name,
                vec![origin_2d, camera.project(&tip)],
                vec![[0.0, 0.0, 0.0], tip],
                color,
            ));
        }
        sidecar
    }
}

fn draw_bloch_vectors_3d(
    latest: &MriKSpaceSnapshot,
    plot_ui: &mut egui_plot::PlotUi<'_>,
    camera: &crate::framework::Camera3D,
    palette: &crate::app::SemanticPalette,
) {
    let proj = |p: [f64; 3]| {
        let [x, y] = camera.project(&p);
        PlotPoint::new(x, y)
    };
    let origin = proj([0.0, 0.0, 0.0]);
    let net = latest.selected_net_m;
    let net_end = proj([net.x, net.y, net.z]);

    for (name, tip, color) in [
        ("Mx", [1.15, 0.0, 0.0], palette.axis_x),
        ("My", [0.0, 1.15, 0.0], palette.axis_y),
        ("Mz", [0.0, 0.0, 1.15], palette.axis_z),
    ] {
        plot_ui.line(Line::new(name, PlotPoints::Owned(vec![origin, proj(tip)])).color(color));
    }
    for i in 0..ISOCHROMAT_COUNT {
        let s = latest.selected_spins[i];
        let pts = PlotPoints::Owned(vec![origin, proj([s.x, s.y, s.z])]);
        plot_ui.line(Line::new("", pts).color(palette.warning).width(1.5_f32));
    }
    if latest.selected_trajectory.len() > 1 {
        let trail: Vec<PlotPoint> = latest
            .selected_trajectory
            .iter()
            .map(|m| proj([m.x, m.y, m.z]))
            .collect();
        plot_ui.line(Line::new("Trajectory", PlotPoints::Owned(trail)).color(palette.axis_y));
    }
    let net_pts = PlotPoints::Owned(vec![origin, net_end]);
    plot_ui.line(
        Line::new("Net M", net_pts)
            .color(palette.accent)
            .width(3.0_f32),
    );
    let tip_pts = PlotPoints::Owned(vec![net_end]);
    plot_ui.points(
        Points::new("", tip_pts)
            .radius(5.0_f32)
            .color(palette.accent),
    );
}

fn project_sphere_ring(camera: &crate::framework::Camera3D, axis: usize) -> PlotPoints<'static> {
    let mut pts = Vec::with_capacity(61);
    for i in 0..=60 {
        let theta = (i as f64) * std::f64::consts::TAU / 60.0;
        let p3 = match axis {
            0 => [theta.cos(), theta.sin(), 0.0],
            1 => [0.0, theta.cos(), theta.sin()],
            _ => [theta.cos(), 0.0, theta.sin()],
        };
        let [x, y] = camera.project(&p3);
        pts.push(PlotPoint::new(x, y));
    }
    PlotPoints::Owned(pts)
}
