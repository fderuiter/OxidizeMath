//! Analytical 2D Turing and Gray-Scott bifurcation regime minimap and CFL sidebar controls.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MorphogenesisTool;
use crate::app::{SemanticPalette, ThemeMode};
use crate::presets::MorphogenesisKineticsMode;
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Stroke};
pub use math_explorer::biology::morphogenesis::{
    classify_gray_scott, classify_schnakenberg, gray_scott_fold_k, gray_scott_hopf_f,
    FusedEulerSolver, GrayScottMinimapRegime, SchnakenbergRegime,
};
use std::sync::atomic::Ordering;

/// Axis range for Schnakenberg activator rate `a`.
pub const SCHNAKENBERG_A_RANGE: (f64, f64) = (0.0, 1.0);
/// Axis range for Schnakenberg inhibitor rate `b`.
pub const SCHNAKENBERG_B_RANGE: (f64, f64) = (0.0, 2.0);
/// Axis range for Gray-Scott substrate feed rate `F`.
pub const GRAY_SCOTT_F_RANGE: (f64, f64) = (0.005, 0.085);
/// Axis range for Gray-Scott catalyst kill rate `k`.
pub const GRAY_SCOTT_K_RANGE: (f64, f64) = (0.030, 0.070);

const ATLAS_TEX_W: usize = 96;
const ATLAS_TEX_H: usize = 72;

/// Cached texture state for the sidebar 2D stability minimap.
#[derive(Default)]
pub struct AtlasCache {
    key: Option<(MorphogenesisKineticsMode, u64, u64, ThemeMode)>,
    texture: Option<egui::TextureHandle>,
}

/// Builds the rasterized `ColorImage` for the active kinetics 2D stability minimap.
#[cfg(test)]
#[must_use]
pub fn build_atlas_image(
    kinetics: MorphogenesisKineticsMode,
    d_u: f64,
    d_v: f64,
    high_contrast: bool,
) -> ColorImage {
    let mode = if high_contrast {
        ThemeMode::HighContrast
    } else {
        ThemeMode::Dark
    };
    build_atlas_image_for_theme(kinetics, d_u, d_v, mode)
}

pub(crate) fn build_atlas_image_for_theme(
    kinetics: MorphogenesisKineticsMode,
    d_u: f64,
    d_v: f64,
    mode: ThemeMode,
) -> ColorImage {
    let palette = mode.palette();
    let mut pixels = Vec::with_capacity(ATLAS_TEX_W * ATLAS_TEX_H);
    let (xr, yr) = atlas_ranges(kinetics);
    for row in 0..ATLAS_TEX_H {
        let ty = 1.0 - ((row as f64 + 0.5) / (ATLAS_TEX_H as f64));
        let y_val = yr.0 + ty * (yr.1 - yr.0);
        for col in 0..ATLAS_TEX_W {
            let tx = (col as f64 + 0.5) / (ATLAS_TEX_W as f64);
            let x_val = xr.0 + tx * (xr.1 - xr.0);
            pixels.push(regime_color(
                kinetics,
                (x_val, y_val),
                (d_u, d_v),
                palette,
                mode,
            ));
        }
    }
    ColorImage {
        size: [ATLAS_TEX_W, ATLAS_TEX_H],
        pixels,
        ..Default::default()
    }
}

fn blend_surface(base: Color32, tint: Color32, weight: u8) -> Color32 {
    let w = u16::from(weight);
    let inv = 255_u16 - w;
    let r = ((u16::from(base.r()) * inv + u16::from(tint.r()) * w) / 255) as u8;
    let g = ((u16::from(base.g()) * inv + u16::from(tint.g()) * w) / 255) as u8;
    let b = ((u16::from(base.b()) * inv + u16::from(tint.b()) * w) / 255) as u8;
    Color32::from_rgba_unmultiplied(r, g, b, 255)
}

fn regime_color(
    kinetics: MorphogenesisKineticsMode,
    coords: (f64, f64),
    diff: (f64, f64),
    palette: SemanticPalette,
    mode: ThemeMode,
) -> Color32 {
    let weight = if mode == ThemeMode::HighContrast {
        195
    } else {
        135
    };
    match kinetics {
        MorphogenesisKineticsMode::Schnakenberg => {
            match classify_schnakenberg(coords.0, coords.1, diff.0, diff.1) {
                SchnakenbergRegime::TuringUnstable => {
                    blend_surface(palette.surface_fill, palette.success, weight)
                }
                SchnakenbergRegime::HopfOscillatory => {
                    blend_surface(palette.surface_fill, palette.warning, weight)
                }
                SchnakenbergRegime::UniformStable => palette.surface_fill,
            }
        }
        MorphogenesisKineticsMode::GrayScott => {
            match classify_gray_scott(coords.0, coords.1, diff.0, diff.1) {
                GrayScottMinimapRegime::BifurcationUnstable => {
                    blend_surface(palette.surface_fill, palette.info, weight)
                }
                GrayScottMinimapRegime::SubcriticalPattern => {
                    blend_surface(palette.surface_fill, palette.warning, weight)
                }
                GrayScottMinimapRegime::UniformStable => palette.surface_fill,
            }
        }
    }
}

/// Returns `(x_range, y_range)` for the active kinetics minimap.
#[must_use]
pub const fn atlas_ranges(kinetics: MorphogenesisKineticsMode) -> ((f64, f64), (f64, f64)) {
    match kinetics {
        MorphogenesisKineticsMode::Schnakenberg => (SCHNAKENBERG_A_RANGE, SCHNAKENBERG_B_RANGE),
        MorphogenesisKineticsMode::GrayScott => (GRAY_SCOTT_F_RANGE, GRAY_SCOTT_K_RANGE),
    }
}

fn param_to_screen(rect: Rect, x: f64, y: f64, xr: (f64, f64), yr: (f64, f64)) -> Pos2 {
    let tx = ((x - xr.0) / (xr.1 - xr.0)).clamp(0.0, 1.0) as f32;
    let ty = ((y - yr.0) / (yr.1 - yr.0)).clamp(0.0, 1.0) as f32;
    egui::pos2(
        rect.left() + tx * rect.width(),
        rect.bottom() - ty * rect.height(),
    )
}

fn screen_to_param(rect: Rect, pos: Pos2, xr: (f64, f64), yr: (f64, f64)) -> (f64, f64) {
    let tx = f64::from(((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0));
    let ty = f64::from(((rect.bottom() - pos.y) / rect.height()).clamp(0.0, 1.0));
    (xr.0 + tx * (xr.1 - xr.0), yr.0 + ty * (yr.1 - yr.0))
}

fn refresh_atlas_texture(
    ctx: &egui::Context,
    cache: &mut AtlasCache,
    kinetics: MorphogenesisKineticsMode,
    d_u: f64,
    d_v: f64,
    mode: ThemeMode,
) {
    let key = (kinetics, d_u.to_bits(), d_v.to_bits(), mode);
    if cache.key == Some(key) && cache.texture.is_some() {
        return;
    }
    let image = build_atlas_image_for_theme(kinetics, d_u, d_v, mode);
    let opts = egui::TextureOptions::LINEAR;
    if let Some(tex) = &mut cache.texture {
        tex.set(image, opts);
    } else {
        cache.texture = Some(ctx.load_texture("morpho_minimap_tex", image, opts));
    }
    cache.key = Some(key);
}

fn paint_atlas_overlays(
    painter: &egui::Painter,
    rect: Rect,
    kinetics: MorphogenesisKineticsMode,
    coords: (f64, f64),
    palette: SemanticPalette,
) -> ((f64, f64), (f64, f64)) {
    painter.rect_stroke(
        rect,
        2.0,
        Stroke::new(1.0_f32, palette.graph_edge),
        egui::StrokeKind::Inside,
    );
    let (xr, yr) = atlas_ranges(kinetics);
    if kinetics == MorphogenesisKineticsMode::GrayScott {
        draw_gray_scott_curves(painter, rect, xr, yr, palette);
    }
    draw_preset_landmarks(painter, rect, kinetics, xr, yr, palette);
    let cross = param_to_screen(rect, coords.0, coords.1, xr, yr);
    painter.circle_stroke(cross, 5.0, Stroke::new(2.0_f32, palette.node_stroke));
    painter.circle_filled(cross, 2.0, palette.error);
    (xr, yr)
}

/// Renders the clickable 2D regime stability minimap and returns updated `(x, y)` coordinates on click/drag.
pub fn draw_regime_atlas(
    ui: &mut egui::Ui,
    cache: &mut AtlasCache,
    kinetics: MorphogenesisKineticsMode,
    coords: (f64, f64),
    diff: (f64, f64),
    high_contrast: bool,
) -> Option<(f64, f64)> {
    let mode = if high_contrast {
        ThemeMode::HighContrast
    } else {
        ThemeMode::from_visuals(ui.visuals())
    };
    let palette = mode.palette();
    refresh_atlas_texture(ui.ctx(), cache, kinetics, diff.0, diff.1, mode);

    let title = match kinetics {
        MorphogenesisKineticsMode::Schnakenberg => "2D Turing Stability Map (a, b)",
        MorphogenesisKineticsMode::GrayScott => "2D Gray-Scott Bifurcation Map (F, k)",
    };
    ui.label(egui::RichText::new(title).strong());
    let width = ui.available_width().clamp(150.0, 240.0);
    let height = (width * 0.68).round();
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), title)
    });

    if let Some(tex) = &mut cache.texture {
        ui.painter().image(
            tex.id(),
            rect,
            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }
    let (xr, yr) = paint_atlas_overlays(ui.painter(), rect, kinetics, coords, palette);
    let caption = match kinetics {
        MorphogenesisKineticsMode::Schnakenberg => "Green: Turing unstable | Amber: Hopf",
        MorphogenesisKineticsMode::GrayScott => "Blue: Fold/Hopf | Amber: Mitosis/Solitons",
    };
    ui.small(caption);

    if response.clicked() || response.dragged() {
        return response
            .interact_pointer_pos()
            .map(|pos| screen_to_param(rect, pos, xr, yr));
    }
    None
}

fn draw_gray_scott_curves(
    painter: &egui::Painter,
    rect: Rect,
    xr: (f64, f64),
    yr: (f64, f64),
    palette: SemanticPalette,
) {
    let mut fold_pts = Vec::with_capacity(49);
    for i in 0..=48 {
        let f = xr.0 + (i as f64 / 48.0) * (0.0625 - xr.0);
        if let Some(k) = gray_scott_fold_k(f) {
            if (yr.0..=yr.1).contains(&k) {
                fold_pts.push(param_to_screen(rect, f, k, xr, yr));
            }
        }
    }
    if fold_pts.len() >= 2 {
        painter.add(egui::Shape::line(
            fold_pts,
            Stroke::new(1.5_f32, palette.accent),
        ));
    }

    let mut hopf_pts = Vec::with_capacity(49);
    for i in 0..=48 {
        let k = yr.0 + (i as f64 / 48.0) * (0.0625 - yr.0);
        if let Some(f) = gray_scott_hopf_f(k) {
            if (xr.0..=xr.1).contains(&f) {
                hopf_pts.push(param_to_screen(rect, f, k, xr, yr));
            }
        }
    }
    if hopf_pts.len() >= 2 {
        painter.add(egui::Shape::line(
            hopf_pts,
            Stroke::new(1.2_f32, palette.info),
        ));
    }
}

fn draw_preset_landmarks(
    painter: &egui::Painter,
    rect: Rect,
    kinetics: MorphogenesisKineticsMode,
    xr: (f64, f64),
    yr: (f64, f64),
    palette: SemanticPalette,
) {
    let pts: &[(f64, f64)] = match kinetics {
        MorphogenesisKineticsMode::Schnakenberg => &[(0.1, 0.9), (0.14, 0.86)],
        MorphogenesisKineticsMode::GrayScott => {
            &[(0.0367, 0.0649), (0.0545, 0.062), (0.030, 0.060)]
        }
    };
    for &(px, py) in pts {
        painter.circle_filled(
            param_to_screen(rect, px, py, xr, yr),
            2.5,
            palette.node_stroke,
        );
    }
}

impl MorphogenesisTool {
    pub(crate) fn apply_minimap_selection(
        &mut self,
        mode: MorphogenesisKineticsMode,
        nx: f64,
        ny: f64,
    ) {
        match mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
                p.a = nx;
                p.b = ny;
            }
            MorphogenesisKineticsMode::GrayScott => {
                let mut ks = self
                    .kinetics_state
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                ks.f = nx;
                ks.k = ny;
            }
        }
        let _ = self.reseed_if_washed_out_entering_pattern();
        self.dirty = true;
    }

    pub(crate) fn draw_cfl_and_minimap(&mut self, ui: &mut egui::Ui) {
        let ks = *self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let (a, b, d_u, d_v, dt, hc) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.a, p.b, p.d_u, p.d_v, p.dt, p.high_contrast)
        };
        let adv_enabled = self
            .advection_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .enabled;
        if !adv_enabled {
            let mut auto_dt = self.auto_dt.load(Ordering::Relaxed);
            let cfl_limit = FusedEulerSolver::cfl_dt_2d(1.0, d_u, d_v);
            let (substeps, eff_dt) = FusedEulerSolver::cfl_substeps_2d(
                dt,
                1.0,
                d_u,
                d_v,
                super::runner::MAX_CFL_SUBSTEPS,
            );
            if ui
                .checkbox(&mut auto_dt, "Auto CFL Substepping")
                .on_hover_text("Automatically splits dt into 2D Forward Euler CFL-safe substeps.")
                .changed()
            {
                self.auto_dt.store(auto_dt, Ordering::Relaxed);
            }
            if auto_dt {
                ui.small(format!(
                    "CFL dt <= {cfl_limit:.4} ({substeps}x {eff_dt:.4})"
                ));
            } else {
                ui.small(format!("Raw Euler dt = {dt:.4} (CFL <= {cfl_limit:.4})"));
            }
        }
        let coords = match ks.mode {
            MorphogenesisKineticsMode::Schnakenberg => (a, b),
            MorphogenesisKineticsMode::GrayScott => (ks.f, ks.k),
        };
        if let Some((nx, ny)) =
            draw_regime_atlas(ui, &mut self.minimap_cache, ks.mode, coords, (d_u, d_v), hc)
        {
            self.apply_minimap_selection(ks.mode, nx, ny);
        }
    }
}
