#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Central workspace layout combining 3D Flamm/wormhole `Plot3DViewport`, 2D gravitational
//! lensing camera, `PhotonFan` ray bundles, twin worldlines, and the 2D parameter regime atlas.

use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::camera::{Plot3DViewport, CAMERA_3D_HELPER_TEXT};
use crate::framework::scoped_panel_id;
use eframe::egui;
use egui_plot::{Line, LineStyle, PlotPoints, Points};

use super::{GeodesicInteractionMode, GeodesicPlaygroundTool, ViewportViewMode};

impl GeodesicPlaygroundTool {
    pub(super) fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.atlas.is_visible() {
                ui.columns(2, |cols| {
                    self.render_active_viewport(&mut cols[0]);
                    self.render_relativity_atlas_panel(&mut cols[1]);
                });
            } else {
                self.render_active_viewport(ui);
            }
        });
    }

    fn render_active_viewport(&mut self, ui: &mut egui::Ui) {
        match self.state.view_mode {
            ViewportViewMode::Embedding3D => self.render_3d_viewport_inside(ui),
            ViewportViewMode::LensingCamera2D => self.draw_lensing_2d_viewport(ui),
        }
    }

    pub(super) fn render_3d_viewport_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let wireframe = self.build_embedding_wireframe();
        let ref_rings = self.build_3d_reference_rings(&palette);
        let twin_proj = (!self.twin_samples.is_empty() && self.active_ray_count <= 1)
            .then(|| self.projected_worldline(&self.twin_samples, self.active_twin_sample()));
        let sidecar = self.build_relativity_3d_sidecar(&palette);

        let mut camera = self.state.camera;
        let lock_orbit = self.locks_camera_orbit();
        let helper = self.viewport_3d_helper_text();
        let plot_id = scoped_panel_id(ui, "relativity_flamm_3d");
        let response = Plot3DViewport::new(plot_id, 16.0)
            .title("Relativity Flamm 3D")
            .helper_text(helper)
            .show(ui, &mut camera, |plot_ui| {
                for (idx, ring) in wireframe.into_iter().enumerate() {
                    plot_ui.line(
                        Line::new(format!("Embedding Mesh {idx}"), PlotPoints::new(ring))
                            .color(palette.graph_edge),
                    );
                }
                for line in ref_rings {
                    plot_ui.line(line);
                }
                self.draw_bundle_worldlines_3d(plot_ui, &palette);
                if let Some((t_line, t_head)) = twin_proj {
                    Self::draw_twin_worldline_3d(plot_ui, &palette, t_line, t_head);
                }
                let _ = self.interact_3d_slingshot(plot_ui, &palette);
                plot_ui.set_3d_sidecar(sidecar);
            })
            .response;

        if response.clicked() || response.drag_started() {
            response.request_focus();
        }
        self.focused_canvas = response.has_focus().then_some(response.id);
        if !lock_orbit && camera != self.state.camera {
            self.state.camera = camera;
            self.revision = self.revision.wrapping_add(1);
        }
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);
        let can_key = response.has_focus() || (response.hovered() && !ctx.wants_keyboard_input());
        let _ = self.handle_keyboard_slingshot(&ctx, can_key && self.locks_camera_orbit());
        response.accessible_theory_hover(self);
    }

    fn build_relativity_3d_sidecar(
        &self,
        palette: &SemanticPalette,
    ) -> egui_plot::Plot3DSeriesSidecar {
        let mut sidecar = egui_plot::Plot3DSeriesSidecar::new();
        let cam = &self.state.camera;
        for (name, world, color) in self.reference_rings_world(palette) {
            let proj = world.iter().map(|p| cam.project(p)).collect();
            sidecar.push_series(egui_plot::Plot3DSeries::new(name, proj, world, color));
        }
        let mut mesh_proj = Vec::new();
        let mut mesh_world = Vec::new();
        for ring in self.build_embedding_wireframe_world() {
            for p in ring {
                mesh_proj.push(cam.project(&p));
                mesh_world.push(p);
            }
        }
        sidecar.push_series(egui_plot::Plot3DSeries::new(
            "Embedding Surface",
            mesh_proj,
            mesh_world,
            palette.graph_edge,
        ));
        self.append_worldline_3d_series(&mut sidecar, palette);
        sidecar
    }

    fn append_worldline_3d_series(
        &self,
        sidecar: &mut egui_plot::Plot3DSeriesSidecar,
        palette: &SemanticPalette,
    ) {
        let cam = &self.state.camera;
        let count = self.active_ray_count.max(1);
        for (idx, ray) in self.bundle_rays.iter().take(count).enumerate() {
            let limit = self
                .state
                .scrub_index
                .saturating_add(1)
                .min(ray.samples.len());
            if limit == 0 {
                continue;
            }
            let world: Vec<[f64; 3]> = ray.samples[..limit].iter().map(|s| s.embed_xyz).collect();
            let proj: Vec<[f64; 2]> = world.iter().map(|p| cam.project(p)).collect();
            let name = if count == 1 {
                "Geodesic Worldline".to_owned()
            } else {
                format!("Geodesic Ray {idx}")
            };
            sidecar.push_series(egui_plot::Plot3DSeries::new(
                name,
                proj,
                world,
                palette.series_color(idx % 6),
            ));
        }
        if !self.twin_samples.is_empty() && count <= 1 {
            let limit = self
                .state
                .scrub_index
                .saturating_add(1)
                .min(self.twin_samples.len());
            let world: Vec<[f64; 3]> = self.twin_samples[..limit]
                .iter()
                .map(|s| s.embed_xyz)
                .collect();
            let proj: Vec<[f64; 2]> = world.iter().map(|p| cam.project(p)).collect();
            sidecar.push_series(egui_plot::Plot3DSeries::new(
                "Twin Worldline",
                proj,
                world,
                palette.series_color(1),
            ));
        }
    }

    fn draw_twin_worldline_3d(
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
        t_line: Vec<[f64; 2]>,
        t_head: Option<[f64; 2]>,
    ) {
        plot_ui.line(
            Line::new("Twin Worldline", PlotPoints::new(t_line))
                .color(palette.series_color(1))
                .width(2.0_f32)
                .style(LineStyle::Dashed { length: 7.0 }),
        );
        if let Some(tp) = t_head {
            plot_ui.points(
                Points::new("Twin Ray", vec![tp])
                    .radius(4.5_f32)
                    .color(palette.series_color(1)),
            );
        }
    }

    const fn viewport_3d_helper_text(&self) -> &'static str {
        match self.state.interaction_mode {
            GeodesicInteractionMode::Orbit => CAMERA_3D_HELPER_TEXT,
            GeodesicInteractionMode::Slingshot => {
                "Slingshot: drag aim vector or press Arrow keys + Space/Enter to launch probe (Camera3D orbit locked)."
            }
            GeodesicInteractionMode::PhotonFan => {
                "Photon Fan: drag aim vector or press Arrow keys + Space/Enter to fan up to 16 geodesics (Camera3D orbit locked)."
            }
        }
    }

    fn draw_bundle_worldlines_3d(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
    ) {
        let count = self.active_ray_count.max(1);
        for (idx, ray) in self.bundle_rays.iter().take(count).enumerate() {
            let limit = self
                .state
                .scrub_index
                .saturating_add(1)
                .min(ray.samples.len());
            if limit == 0 {
                continue;
            }
            let pts: Vec<[f64; 2]> = ray.samples[..limit]
                .iter()
                .map(|s| self.state.camera.project(&s.embed_xyz))
                .collect();
            let (name, width) = if count == 1 {
                ("Geodesic Worldline".to_string(), 2.4_f32)
            } else {
                (format!("Geodesic Ray {idx}"), 1.7_f32)
            };
            plot_ui.line(
                Line::new(name, PlotPoints::new(pts))
                    .color(palette.series_color(idx % 6))
                    .width(width),
            );
            if let Some(last) = ray.samples.get(limit - 1) {
                let head = self.state.camera.project(&last.embed_xyz);
                let head_name = if count == 1 {
                    "Particle / Photon".to_string()
                } else {
                    format!("Head {idx}")
                };
                plot_ui.points(
                    Points::new(head_name, vec![head])
                        .radius(if count == 1 { 5.0_f32 } else { 4.5_f32 })
                        .color(palette.accent),
                );
            }
        }
    }
}
