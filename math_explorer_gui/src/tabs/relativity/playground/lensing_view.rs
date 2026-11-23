#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 2D deflection-table gravitational lensing camera warping galaxies, Lenia solitons, and starfields.

use crate::accessibility::{
    paint_canvas_focus_ring, register_canvas_accessibility, AccessibleTheoryHover,
};
use crate::app::SemanticPalette;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::high_energy::{
    LensedRayHit, LensingMetricSpec, EINSTEIN_RING_IMPACT_FACTOR,
};

use super::{
    GeodesicInteractionMode, GeodesicPlaygroundTool, LensingBackgroundKind, SpacetimePreset,
    LENSING_HEIGHT, LENSING_WIDTH,
};

impl GeodesicPlaygroundTool {
    /// Advances cross-domain lensing sources (`GalacticDiskSystem` and `LeniaEngine`) with zero heap allocations.
    pub fn step_lensing_backgrounds(&mut self) {
        self.lensing_galaxy.step_leapfrog(0.65);
        #[cfg(feature = "biology")]
        {
            self.lenia_engine.step();
        }
        self.refresh_lensing_pixels_in_place();
    }

    /// Rebuilds the `DeflectionTable` in place when spacetime parameters change and warps the
    /// `LENSING_WIDTH x LENSING_HEIGHT` raster buffer with zero heap allocations.
    pub fn refresh_lensing_pixels_in_place(&mut self) {
        if self.lensing_dirty {
            let spec = self.current_lensing_spec();
            let fov_b = self.lensing_fov_b_max();
            let cam_dist = (10.0 * self.state.scale_radius).max(16.0);
            let _ = self
                .deflection_table
                .rebuild_in_place(spec, cam_dist, fov_b * 1.35);
            self.lensing_dirty = false;
        }
        let fov_b = self.lensing_fov_b_max();
        let w = LENSING_WIDTH;
        let h = LENSING_HEIGHT;
        let aspect = (w as f64) / (h as f64);
        for py in 0..h {
            let v = 1.0 - 2.0 * ((py as f64 + 0.5) / (h as f64));
            for px in 0..w {
                let u = (-1.0 + 2.0 * ((px as f64 + 0.5) / (w as f64))) * aspect;
                let hit = self.deflection_table.map_image_uv(u, v, fov_b);
                let idx = flatten_2d_index(px, py, w);
                if idx < self.lensing_pixels.len() {
                    self.lensing_pixels[idx] = self.shade_lensed_hit(&hit);
                }
            }
        }
    }

    /// Clones the pre-rendered lensed raster buffer into an [`egui::ColorImage`] for PNG/SVG export,
    /// computing an on-demand frame if the 2D lensing camera has not yet been rendered.
    pub fn build_lensing_color_image(&self) -> egui::ColorImage {
        let pixels = if self.lensing_dirty {
            self.render_on_demand_lensing_pixels()
        } else {
            self.lensing_pixels.clone()
        };
        egui::ColorImage {
            size: [LENSING_WIDTH, LENSING_HEIGHT],
            pixels,
            source_size: egui::vec2(LENSING_WIDTH as f32, LENSING_HEIGHT as f32),
        }
    }

    fn render_on_demand_lensing_pixels(&self) -> Vec<egui::Color32> {
        let spec = self.current_lensing_spec();
        let fov_b = self.lensing_fov_b_max();
        let cam_dist = (10.0 * self.state.scale_radius).max(16.0);
        let Ok(table) = math_explorer::physics::high_energy::DeflectionTable::build(
            spec,
            cam_dist,
            fov_b * 1.35,
            65,
        ) else {
            return self.lensing_pixels.clone();
        };
        let w = LENSING_WIDTH;
        let h = LENSING_HEIGHT;
        let aspect = (w as f64) / (h as f64);
        let mut out = vec![egui::Color32::BLACK; w * h];
        for py in 0..h {
            let v = 1.0 - 2.0 * ((py as f64 + 0.5) / (h as f64));
            for px in 0..w {
                let u = (-1.0 + 2.0 * ((px as f64 + 0.5) / (w as f64))) * aspect;
                let hit = table.map_image_uv(u, v, fov_b);
                let idx = flatten_2d_index(px, py, w);
                out[idx] = self.shade_lensed_hit(&hit);
            }
        }
        out
    }

    pub(super) fn current_lensing_spec(&self) -> LensingMetricSpec {
        let rs = self.state.scale_radius.clamp(0.5, 5.0);
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild => LensingMetricSpec::Schwarzschild {
                schwarzschild_radius: rs,
            },
            SpacetimePreset::CustomRadialLapse => LensingMetricSpec::Schwarzschild {
                schwarzschild_radius: self.sandbox.estimate_lapse_horizon(rs).clamp(0.5, 5.0),
            },
            SpacetimePreset::KerrEquatorial => LensingMetricSpec::KerrEquatorial {
                schwarzschild_radius: rs,
                spin_a: 0.5 * rs * self.state.spin_chi.clamp(-0.95, 0.95),
            },
            SpacetimePreset::MorrisThorneWormhole | SpacetimePreset::CustomWormholeProfile => {
                LensingMetricSpec::MorrisThorne { throat_radius: rs }
            }
        }
    }

    pub(super) fn lensing_fov_b_max(&self) -> f64 {
        (5.2 * self.state.scale_radius.max(0.8)).max(6.5)
    }

    fn shade_lensed_hit(&self, hit: &LensedRayHit) -> egui::Color32 {
        if hit.captured {
            return if self.high_contrast {
                egui::Color32::BLACK
            } else {
                egui::Visuals::dark().extreme_bg_color
            };
        }
        let mut rgb = if hit.sheet_sign < 0
            || self.state.lensing_background == LensingBackgroundKind::LeniaPortal
        {
            self.sample_portal_sheet_rgb(hit.source_u, hit.source_v, hit.sheet_sign < 0)
        } else {
            match self.state.lensing_background {
                LensingBackgroundKind::GalacticDisk => {
                    let kpc_scale = 3.8 * self.lensing_galaxy.baryons.disk_scale_kpc.max(0.5);
                    let mut gal = self.lensing_galaxy.sample_surface_brightness(
                        hit.source_u * kpc_scale,
                        hit.source_v * kpc_scale,
                    );
                    let star = sample_stargrid(hit.source_u, hit.source_v);
                    gal[0] = (gal[0] + 0.35 * star).clamp(0.0, 1.0);
                    gal[1] = (gal[1] + 0.38 * star).clamp(0.0, 1.0);
                    gal[2] = (gal[2] + 0.45 * star).clamp(0.0, 1.0);
                    gal
                }
                LensingBackgroundKind::Starfield | LensingBackgroundKind::LeniaPortal => {
                    let s = sample_stargrid(hit.source_u, hit.source_v);
                    let neb = 0.14 * (3.0 * hit.source_u).sin().abs();
                    [
                        (0.10 + 0.85 * s + neb).clamp(0.0, 1.0),
                        (0.14 + 0.88 * s + 0.6 * neb).clamp(0.0, 1.0),
                        (0.24 + 0.95 * s).clamp(0.0, 1.0),
                    ]
                }
            }
        };

        let ring = (hit.ring_boost * hit.redshift_factor.clamp(0.4, 1.8)).clamp(0.0, 1.0);
        rgb[0] = (rgb[0] + 0.92 * ring).clamp(0.0, 1.0);
        rgb[1] = (rgb[1] + 0.76 * ring).clamp(0.0, 1.0);
        rgb[2] = (rgb[2] + 0.38 * ring).clamp(0.0, 1.0);
        let base = egui::Color32::from(egui::Rgba::from_rgb(
            rgb[0] as f32,
            rgb[1] as f32,
            rgb[2] as f32,
        ));
        if self.high_contrast {
            let [r, g, b] = math_commons::math_kernel::colormap::ensure_dark_canvas_contrast([
                base.r(),
                base.g(),
                base.b(),
            ]);
            egui::Color32::from_rgb(r, g, b)
        } else {
            base
        }
    }

    fn sample_portal_sheet_rgb(&self, u: f64, v: f64, in_throat: bool) -> [f64; 3] {
        #[cfg(feature = "biology")]
        {
            let w = self.lenia_engine.width().max(1);
            let h = self.lenia_engine.height().max(1);
            let gx = ((0.5 + 0.42 * u).rem_euclid(1.0) * (w as f64)) as usize % w;
            let gy = ((0.5 + 0.42 * v).rem_euclid(1.0) * (h as f64)) as usize % h;
            let cell = self
                .lenia_engine
                .state()
                .get(flatten_2d_index(gx, gy, w))
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let star = sample_stargrid(u * 1.3, v * 1.3) * 0.30;
            let tint = if in_throat { 0.18 } else { 0.06 };
            [
                (tint + 0.30 * cell + 0.85 * cell * cell + star).clamp(0.0, 1.0),
                (0.10 + 0.92 * cell + star).clamp(0.0, 1.0),
                (0.22 + 0.78 * cell.sqrt() + star).clamp(0.0, 1.0),
            ]
        }
        #[cfg(not(feature = "biology"))]
        {
            let r = u.hypot(v);
            let ring_soliton = (-16.0 * (r - 0.36).powi(2)).exp();
            let gal = self
                .lensing_galaxy
                .sample_surface_brightness(u * 3.0, v * 3.0);
            let portal_tint = if in_throat { 0.20 } else { 0.05 };
            [
                (portal_tint + 0.55 * gal[0] + 0.65 * ring_soliton).clamp(0.0, 1.0),
                (0.12 + 0.55 * gal[1] + 0.88 * ring_soliton).clamp(0.0, 1.0),
                (0.22 + 0.65 * gal[2] + 0.80 * ring_soliton).clamp(0.0, 1.0),
            ]
        }
    }

    pub(super) fn draw_lensing_2d_viewport(&mut self, ui: &mut egui::Ui) {
        let is_hc = ui.visuals().override_text_color.is_some()
            && ui.visuals().extreme_bg_color == egui::Color32::BLACK;
        if is_hc != self.high_contrast || self.lensing_dirty {
            self.high_contrast = is_hc;
            self.refresh_lensing_pixels_in_place();
        }
        let img = self.build_lensing_color_image();
        let tex = self.lensing_texture.get_or_insert_with(|| {
            ui.ctx().load_texture(
                "relativity_lensing_cam",
                img.clone(),
                egui::TextureOptions::LINEAR,
            )
        });
        tex.set(img, egui::TextureOptions::LINEAR);

        let avail = ui.available_size();
        let target_w = avail.x.max(180.0);
        let target_h = (target_w * 0.75).min(avail.y.max(140.0));
        let draw_size = egui::vec2(target_h * (4.0 / 3.0), target_h).min(avail);
        let (rect, response) = ui.allocate_exact_size(
            draw_size,
            egui::Sense::click_and_drag().union(egui::Sense::focusable_noninteractive()),
        );
        if response.clicked() || response.drag_started() {
            response.request_focus();
        }
        self.focused_canvas = response.has_focus().then_some(response.id);

        ui.painter().image(
            tex.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        let palette = SemanticPalette::from_visuals(ui.visuals());
        self.paint_lensing_critical_overlays(ui, rect, &palette);
        self.interact_lensing_2d_drag(ui, rect, &response, &palette);

        let crit_desc = self.lensing_accessibility_summary();
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::Image,
            "Relativistic 2D Gravitational Lensing Camera",
            Some(&crit_desc),
        );
        paint_canvas_focus_ring(ui, &response);
        response.accessible_theory_hover(self);
    }

    fn paint_lensing_critical_overlays(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let fov_b = self.lensing_fov_b_max() as f32;
        let half_h = 0.5 * rect.height();
        let center = rect.center();
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild | SpacetimePreset::CustomRadialLapse => {
                let r_h = if self.state.spacetime == SpacetimePreset::CustomRadialLapse {
                    self.sandbox
                        .estimate_lapse_horizon(self.state.scale_radius)
                        .max(0.5)
                } else {
                    self.state.scale_radius
                };
                let b_einstein = (EINSTEIN_RING_IMPACT_FACTOR * r_h) as f32;
                let r_px = (b_einstein / fov_b) * half_h;
                ui.painter().circle_stroke(
                    center,
                    r_px,
                    egui::Stroke::new(1.4_f32, palette.warning),
                );
            }
            SpacetimePreset::KerrEquatorial => {
                let (b_neg, b_pos) = self.deflection_table.critical_impact_bounds();
                let x_pro = center.x + ((b_pos as f32) / fov_b) * half_h;
                let x_ret = center.x + ((b_neg as f32) / fov_b) * half_h;
                ui.painter()
                    .circle_filled(egui::pos2(x_pro, center.y), 3.5_f32, palette.accent);
                ui.painter()
                    .circle_filled(egui::pos2(x_ret, center.y), 3.5_f32, palette.warning);
            }
            SpacetimePreset::MorrisThorneWormhole | SpacetimePreset::CustomWormholeProfile => {
                let r_throat = ((self.state.scale_radius as f32) / fov_b) * half_h;
                ui.painter().circle_stroke(
                    center,
                    r_throat,
                    egui::Stroke::new(1.5_f32, palette.info),
                );
            }
        }
    }

    fn interact_lensing_2d_drag(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        response: &egui::Response,
        palette: &SemanticPalette,
    ) {
        crate::accessibility::lock_canvas_arrow_keys(ui, response);
        let ctx = ui.ctx();
        let _ = self.handle_keyboard_slingshot(ctx, response.has_focus());
        if self.state.interaction_mode == GeodesicInteractionMode::Orbit {
            return;
        }
        let to_uv = |p: egui::Pos2| -> [f64; 2] {
            let half_h = (0.5 * rect.height()).max(1.0);
            [
                f64::from((p.x - rect.center().x) / half_h),
                f64::from((rect.center().y - p.y) / half_h),
            ]
        };
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let uv = to_uv(pos);
                self.slingshot.drag_origin = Some(uv);
                self.slingshot.drag_tip = Some(uv);
            }
        }
        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let uv = to_uv(pos);
                let origin = self.slingshot.drag_origin.unwrap_or(uv);
                self.slingshot.drag_origin = Some(origin);
                self.slingshot.drag_tip = Some(uv);
                self.apply_lensing_canvas_drag(origin, uv);
            }
        }
        if response.drag_stopped() {
            self.slingshot.drag_origin = None;
            self.slingshot.drag_tip = None;
        }
        if let (Some(o), Some(t)) = (self.slingshot.drag_origin, self.slingshot.drag_tip) {
            let half_h = 0.5 * rect.height();
            let p0 = egui::pos2(
                rect.center().x + (o[0] as f32) * half_h,
                rect.center().y - (o[1] as f32) * half_h,
            );
            let p1 = egui::pos2(
                rect.center().x + (t[0] as f32) * half_h,
                rect.center().y - (t[1] as f32) * half_h,
            );
            ui.painter()
                .line_segment([p0, p1], egui::Stroke::new(2.0_f32, palette.warning));
        }
    }

    fn lensing_accessibility_summary(&self) -> String {
        let (b_neg, b_pos) = self
            .current_lensing_spec()
            .critical_impact_bounds()
            .unwrap_or_else(|_| self.deflection_table.critical_impact_bounds());
        format!(
            "{} lensing {} with critical bounds [{:.2}, {:.2}]",
            self.state.spacetime.label(),
            self.state.lensing_background.label(),
            b_neg,
            b_pos
        )
    }
}

fn sample_stargrid(u: f64, v: f64) -> f64 {
    let cell_x = (u * 14.0).floor();
    let cell_y = (v * 14.0).floor();
    let fx = (u * 14.0) - cell_x - 0.5;
    let fy = (v * 14.0) - cell_y - 0.5;
    let hash = ((cell_x * 127.1 + cell_y * 311.7).sin() * 43_758.545_3)
        .fract()
        .abs();
    if hash > 0.84 {
        let d2 = fx * fx + fy * fy;
        (-42.0 * d2).exp() * hash
    } else {
        0.0
    }
}
