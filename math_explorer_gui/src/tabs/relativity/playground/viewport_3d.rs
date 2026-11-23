#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D Flamm paraboloid and Morris-Thorne two-sheet `Plot3DViewport` renderer and landmark rings.

use std::f64::consts::TAU;

use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::camera::{Plot3DViewport, CAMERA_3D_HELPER_TEXT};
use crate::framework::scoped_panel_id;
use eframe::egui;
use egui_plot::{Line, LineStyle, PlotPoints, Points};
use math_explorer::physics::high_energy::{
    flamm_paraboloid_z, KerrEquatorialMetric, MorrisThorneWormhole, SchwarzschildBlackHole,
};

use super::{GeodesicInteractionMode, GeodesicPlaygroundTool, SpacetimePreset};

struct LandmarkRings3D {
    horizon: Vec<[f64; 2]>,
    primary: Option<(&'static str, Vec<[f64; 2]>)>,
    secondary: Option<(&'static str, Vec<[f64; 2]>)>,
}

impl GeodesicPlaygroundTool {
    pub(super) fn draw_central_3d_viewport(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let ctx = ui.ctx().clone();
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let wireframe = self.build_embedding_wireframe();
            let rings = self.build_landmark_rings_3d();

            let mut camera = self.state.camera;
            let lock_orbit = self.locks_camera_orbit();
            let helper = self.viewport_3d_helper_text();
            let plot_id = scoped_panel_id(ui, "relativity_flamm_3d");
            let response = Plot3DViewport::new(plot_id, 16.0)
                .helper_text(helper)
                .show(ui, &mut camera, |plot_ui| {
                    for (idx, ring) in wireframe.into_iter().enumerate() {
                        plot_ui.line(
                            Line::new(format!("Embedding Mesh {idx}"), PlotPoints::new(ring))
                                .color(palette.graph_edge),
                        );
                    }
                    plot_ui.line(
                        Line::new("Horizon / Throat", PlotPoints::new(rings.horizon))
                            .color(palette.error),
                    );
                    if let Some((label, ph)) = rings.primary {
                        plot_ui.line(Line::new(label, PlotPoints::new(ph)).color(palette.warning));
                    }
                    if let Some((label, aux)) = rings.secondary {
                        plot_ui.line(
                            Line::new(label, PlotPoints::new(aux))
                                .color(palette.info)
                                .style(LineStyle::Dashed { length: 5.0 }),
                        );
                    }
                    self.draw_bundle_worldlines_3d(plot_ui, &palette);
                    let _ = self.interact_3d_slingshot(plot_ui, &palette);
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
            let can_key =
                response.has_focus() || (response.hovered() && !ctx.wants_keyboard_input());
            let _ = self.handle_keyboard_slingshot(&ctx, can_key);
            response.accessible_theory_hover(self);
        });
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

    fn build_landmark_rings_3d(&self) -> LandmarkRings3D {
        let rs = self.state.scale_radius;
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild => {
                let bh = SchwarzschildBlackHole::from_schwarzschild_radius(rs).ok();
                let r_ph = bh
                    .and_then(|b| b.photon_sphere_radius().ok())
                    .unwrap_or(1.5 * rs);
                let r_isco = bh.and_then(|b| b.isco().ok()).unwrap_or(3.0 * rs);
                let z_ph = flamm_paraboloid_z(rs, r_ph).unwrap_or(0.0);
                let z_isco = flamm_paraboloid_z(rs, r_isco).unwrap_or(0.0);
                LandmarkRings3D {
                    horizon: self.reference_ring_projected(rs, 0.0, 56),
                    primary: Some((
                        "Photon Sphere (1.5 Rs)",
                        self.reference_ring_projected(r_ph, z_ph, 56),
                    )),
                    secondary: Some((
                        "ISCO (3.0 Rs)",
                        self.reference_ring_projected(r_isco, z_isco, 56),
                    )),
                }
            }
            SpacetimePreset::KerrEquatorial => {
                let a = 0.5 * rs * self.state.spin_chi;
                if let Ok(kerr) = KerrEquatorialMetric::new(rs, a) {
                    let r_plus = kerr.outer_horizon_radius();
                    let r_ergo = kerr.ergosphere_radius();
                    let r_ph = kerr.photon_orbit_radius(true);
                    let z_ergo = flamm_paraboloid_z(r_plus, r_ergo).unwrap_or(0.0);
                    let z_ph = flamm_paraboloid_z(r_plus, r_ph.max(r_plus)).unwrap_or(0.0);
                    return LandmarkRings3D {
                        horizon: self.reference_ring_projected(r_plus, 0.0, 56),
                        primary: Some((
                            "Prograde Photon Orbit",
                            self.reference_ring_projected(r_ph, z_ph, 56),
                        )),
                        secondary: Some((
                            "Ergosphere Boundary",
                            self.reference_ring_projected(r_ergo, z_ergo, 56),
                        )),
                    };
                }
                LandmarkRings3D {
                    horizon: self.reference_ring_projected(rs, 0.0, 56),
                    primary: None,
                    secondary: None,
                }
            }
            SpacetimePreset::MorrisThorneWormhole => LandmarkRings3D {
                horizon: self.reference_ring_projected(rs, 0.0, 56),
                primary: None,
                secondary: None,
            },
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
            let color = palette.series_color(idx % 6);
            plot_ui.line(
                Line::new(format!("Geodesic Ray {idx}"), PlotPoints::new(pts))
                    .color(color)
                    .width(if count == 1 { 2.4_f32 } else { 1.7_f32 }),
            );
            if let Some(last) = ray.samples.get(limit - 1) {
                let head = self.state.camera.project(&last.embed_xyz);
                plot_ui.points(
                    Points::new(format!("Head {idx}"), vec![head])
                        .radius(4.5_f32)
                        .color(palette.accent),
                );
            }
        }
    }

    fn build_embedding_wireframe(&self) -> Vec<Vec<[f64; 2]>> {
        let mut lines = Vec::with_capacity(14);
        let rs = self.state.scale_radius;
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild | SpacetimePreset::KerrEquatorial => {
                let r_base = if self.state.spacetime == SpacetimePreset::KerrEquatorial {
                    KerrEquatorialMetric::new(rs, 0.5 * rs * self.state.spin_chi)
                        .map(|k| k.outer_horizon_radius())
                        .unwrap_or(rs)
                } else {
                    rs
                };
                for ring_idx in 1..=5 {
                    let r = r_base * (1.0 + 1.35 * f64::from(ring_idx));
                    let z = flamm_paraboloid_z(r_base, r).unwrap_or(0.0);
                    lines.push(self.reference_ring_projected(r, z, 40));
                }
                for spoke in 0..8 {
                    let phi = TAU * f64::from(spoke) / 8.0;
                    let (s, c) = phi.sin_cos();
                    let mut meridian = Vec::with_capacity(18);
                    for k in 0..18 {
                        let r = r_base + 7.0 * rs * (k as f64) / 17.0;
                        let z = flamm_paraboloid_z(r_base, r).unwrap_or(0.0);
                        meridian.push(self.state.camera.project(&[r * c, r * s, z]));
                    }
                    lines.push(meridian);
                }
            }
            SpacetimePreset::MorrisThorneWormhole => {
                let Ok(wh) = MorrisThorneWormhole::new(rs) else {
                    return lines;
                };
                for level in [-6.0, -3.0, 3.0, 6.0] {
                    let r = wh.areal_radius(level).unwrap_or(rs);
                    let z = wh.embedding_z(level).unwrap_or(0.0);
                    lines.push(self.reference_ring_projected(r, z, 40));
                }
                for spoke in 0..8 {
                    let phi = TAU * f64::from(spoke) / 8.0;
                    let mut meridian = Vec::with_capacity(21);
                    for k in 0..=20 {
                        let l = -8.0 + 16.0 * (k as f64) / 20.0;
                        if let Ok(xyz) = wh.embedding_xyz(l, phi) {
                            meridian.push(self.state.camera.project(&xyz));
                        }
                    }
                    lines.push(meridian);
                }
            }
        }
        lines
    }
}
