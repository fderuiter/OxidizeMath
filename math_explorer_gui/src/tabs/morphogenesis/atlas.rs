//! Analytical 2D Turing and Gray-Scott bifurcation regime atlas and interactive minimap widget.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::MorphogenesisKinetics;
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Stroke};

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

/// Linear stability classification for Schnakenberg `(a, b)` parameters at `(D_u, D_v)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchnakenbergRegime {
    /// Diffusion-driven Turing instability (`tr < 0`, `det > 0`, dispersion threshold exceeded).
    TuringUnstable,
    /// Homogeneous Hopf oscillatory instability (`tr >= 0`, `det > 0`).
    HopfOscillatory,
    /// Uniform stable steady state with no spatial pattern growth.
    UniformStable,
}

/// Bifurcation and pattern-forming classification for Gray-Scott `(F, k)` parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrayScottRegime {
    /// Turing or Hopf pattern-forming instability inside the saddle-node fold `F >= 4(F + k)^2`.
    BifurcationUnstable,
    /// Subcritical excitable pulse/mitosis/soliton band adjacent to the fold curve.
    SubcriticalPattern,
    /// Uniform high-catalyst or trivial substrate-only equilibrium.
    UniformStable,
}

/// Classifies Schnakenberg `(a, b)` stability analytically for diffusion rates `(d_u, d_v)`.
#[must_use]
pub fn classify_schnakenberg(a: f64, b: f64, d_u: f64, d_v: f64) -> SchnakenbergRegime {
    let sum = a + b;
    if !a.is_finite()
        || !b.is_finite()
        || !d_u.is_finite()
        || !d_v.is_finite()
        || a < 0.0
        || b <= 0.0
        || sum <= 1e-9
        || d_u <= 0.0
        || d_v <= 0.0
    {
        return SchnakenbergRegime::UniformStable;
    }
    let f_u = (b - a) / sum;
    let f_v = sum * sum;
    let g_u = -2.0 * b / sum;
    let g_v = -(sum * sum);
    let trace = f_u + g_v;
    let det = f_u * g_v - f_v * g_u;
    if trace < 0.0 && det > 0.0 && (d_v * f_u + d_u * g_v) > 2.0 * (d_u * d_v * det).sqrt() {
        SchnakenbergRegime::TuringUnstable
    } else if trace >= 0.0 && det > 0.0 {
        SchnakenbergRegime::HopfOscillatory
    } else {
        SchnakenbergRegime::UniformStable
    }
}

/// Saddle-node fold curve `k_fold(F) = sqrt(F) / 2 - F` solving `F = 4 * (F + k)^2`.
#[must_use]
pub fn gray_scott_fold_k(f: f64) -> Option<f64> {
    if !f.is_finite() || !(0.0..=0.25).contains(&f) {
        return None;
    }
    Some(0.5 * f.sqrt() - f)
}

/// Hopf curve `F_hopf(k)` solving `(F + k)^2 = F * sqrt(k)` on the upper catalyst branch (`k <= 1/16`).
#[must_use]
pub fn gray_scott_hopf_f(k: f64) -> Option<f64> {
    if !k.is_finite() || !(0.0..=0.0625).contains(&k) {
        return None;
    }
    let sqrt_k = k.sqrt();
    let disc = (k * (1.0 - 4.0 * sqrt_k)).max(0.0);
    Some(0.5 * ((sqrt_k - 2.0 * k) - disc.sqrt()))
}

/// Classifies Gray-Scott `(f, k)` parameters relative to the fold `F = 4(F + k)^2` and Hopf curve.
#[must_use]
pub fn classify_gray_scott(f: f64, k: f64, d_u: f64, d_v: f64) -> GrayScottRegime {
    if !f.is_finite()
        || !k.is_finite()
        || !d_u.is_finite()
        || !d_v.is_finite()
        || f <= 0.0
        || k <= 0.0
        || d_u <= 0.0
        || d_v <= 0.0
    {
        return GrayScottRegime::UniformStable;
    }
    let sum = f + k;
    let fold_rhs = 4.0 * sum * sum;
    if f >= fold_rhs {
        let disc = (f * (f - fold_rhs)).max(0.0).sqrt();
        let v_plus = (f + disc) / (2.0 * sum);
        let f_u = -(v_plus * v_plus) - f;
        let g_v = sum;
        let det = sum * (v_plus * v_plus - f);
        let trace = f_u + g_v;
        let turing = det > 0.0 && (d_v * f_u + d_u * g_v) > 2.0 * (d_u * d_v * det).sqrt();
        if trace >= 0.0 || turing {
            return GrayScottRegime::BifurcationUnstable;
        }
        return GrayScottRegime::UniformStable;
    }
    if let Some(k_fold) = gray_scott_fold_k(f) {
        if (0.018..=0.0625).contains(&f) && k > k_fold && k <= k_fold + 0.0065 {
            return GrayScottRegime::SubcriticalPattern;
        }
    }
    GrayScottRegime::UniformStable
}

/// Cached texture state for the 2D stability atlas minimap.
#[derive(Default)]
pub struct AtlasCache {
    key: Option<(MorphogenesisKinetics, u64, u64, ThemeMode)>,
    texture: Option<egui::TextureHandle>,
}

/// Builds the rasterized `ColorImage` for the active kinetics 2D stability minimap.
#[cfg(test)]
#[must_use]
pub fn build_atlas_image(
    kinetics: MorphogenesisKinetics,
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

fn build_atlas_image_for_theme(
    kinetics: MorphogenesisKinetics,
    d_u: f64,
    d_v: f64,
    mode: ThemeMode,
) -> ColorImage {
    let palette = mode.palette();
    let mut pixels = Vec::with_capacity(ATLAS_TEX_W * ATLAS_TEX_H);
    let (x_range, y_range) = atlas_ranges(kinetics);
    for row in 0..ATLAS_TEX_H {
        let ty = 1.0 - ((row as f64 + 0.5) / (ATLAS_TEX_H as f64));
        let y_val = y_range.0 + ty * (y_range.1 - y_range.0);
        for col in 0..ATLAS_TEX_W {
            let tx = (col as f64 + 0.5) / (ATLAS_TEX_W as f64);
            let x_val = x_range.0 + tx * (x_range.1 - x_range.0);
            pixels.push(regime_color(kinetics, x_val, y_val, d_u, d_v, palette, mode));
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
    kinetics: MorphogenesisKinetics,
    x: f64,
    y: f64,
    d_u: f64,
    d_v: f64,
    palette: SemanticPalette,
    mode: ThemeMode,
) -> Color32 {
    let weight = if mode == ThemeMode::HighContrast {
        195
    } else {
        135
    };
    match kinetics {
        MorphogenesisKinetics::Schnakenberg => match classify_schnakenberg(x, y, d_u, d_v) {
            SchnakenbergRegime::TuringUnstable => {
                blend_surface(palette.surface_fill, palette.success, weight)
            }
            SchnakenbergRegime::HopfOscillatory => {
                blend_surface(palette.surface_fill, palette.warning, weight)
            }
            SchnakenbergRegime::UniformStable => palette.surface_fill,
        },
        MorphogenesisKinetics::GrayScott => match classify_gray_scott(x, y, d_u, d_v) {
            GrayScottRegime::BifurcationUnstable => {
                blend_surface(palette.surface_fill, palette.info, weight)
            }
            GrayScottRegime::SubcriticalPattern => {
                blend_surface(palette.surface_fill, palette.warning, weight)
            }
            GrayScottRegime::UniformStable => palette.surface_fill,
        },
    }
}

/// Returns `(x_range, y_range)` for the active kinetics atlas.
#[must_use]
pub const fn atlas_ranges(kinetics: MorphogenesisKinetics) -> ((f64, f64), (f64, f64)) {
    match kinetics {
        MorphogenesisKinetics::Schnakenberg => (SCHNAKENBERG_A_RANGE, SCHNAKENBERG_B_RANGE),
        MorphogenesisKinetics::GrayScott => (GRAY_SCOTT_F_RANGE, GRAY_SCOTT_K_RANGE),
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
    kinetics: MorphogenesisKinetics,
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
        cache.texture = Some(ctx.load_texture("morpho_atlas_tex", image, opts));
    }
    cache.key = Some(key);
}

/// Renders the clickable 2D regime atlas minimap and returns updated `(x, y)` coordinates on click/drag.
#[allow(clippy::too_many_arguments)]
pub fn draw_regime_atlas(
    ui: &mut egui::Ui,
    cache: &mut AtlasCache,
    kinetics: MorphogenesisKinetics,
    x_val: f64,
    y_val: f64,
    d_u: f64,
    d_v: f64,
    high_contrast: bool,
) -> Option<(f64, f64)> {
    let mode = if high_contrast {
        ThemeMode::HighContrast
    } else {
        ThemeMode::from_visuals(ui.visuals())
    };
    let palette = mode.palette();
    refresh_atlas_texture(ui.ctx(), cache, kinetics, d_u, d_v, mode);

    let title = match kinetics {
        MorphogenesisKinetics::Schnakenberg => "2D Turing Atlas (a, b)",
        MorphogenesisKinetics::GrayScott => "2D Gray-Scott Atlas (F, k)",
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
    ui.painter().rect_stroke(
        rect,
        2.0,
        Stroke::new(1.0_f32, palette.graph_edge),
        egui::StrokeKind::Inside,
    );

    let (xr, yr) = atlas_ranges(kinetics);
    if kinetics == MorphogenesisKinetics::GrayScott {
        draw_gray_scott_curves(ui.painter(), rect, xr, yr, palette);
    }
    draw_preset_landmarks(ui.painter(), rect, kinetics, xr, yr, palette);

    let cross = param_to_screen(rect, x_val, y_val, xr, yr);
    ui.painter()
        .circle_stroke(cross, 5.0, Stroke::new(2.0_f32, palette.node_stroke));
    ui.painter().circle_filled(cross, 2.0, palette.error);

    let caption = match kinetics {
        MorphogenesisKinetics::Schnakenberg => "Green: Turing unstable | Amber: Hopf",
        MorphogenesisKinetics::GrayScott => "Blue: Fold/Hopf | Amber: Mitosis/Solitons",
    };
    ui.small(caption);

    if (response.clicked() || response.dragged()) && response.interact_pointer_pos().is_some() {
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
    kinetics: MorphogenesisKinetics,
    xr: (f64, f64),
    yr: (f64, f64),
    palette: SemanticPalette,
) {
    let pts: &[(f64, f64)] = match kinetics {
        MorphogenesisKinetics::Schnakenberg => &[(0.1, 0.9), (0.14, 0.86)],
        MorphogenesisKinetics::GrayScott => &[(0.0367, 0.0649), (0.0545, 0.062), (0.030, 0.060)],
    };
    for &(px, py) in pts {
        let p = param_to_screen(rect, px, py, xr, yr);
        painter.circle_filled(p, 2.5, palette.node_stroke);
    }
}
