//! Analytical CPU and WGSL/GLSL shader colormap emitters with WCAG 2.2 `3.0:1` contrast lifting.
// [cite:math_commons]

use super::{
    cyclic_cosine_palette, cyclic_cosine_palette_high_contrast, ensure_dark_canvas_contrast,
    heatmap_color, heatmap_color_high_contrast, mapped_gradient,
};
use crate::ast::ShaderTarget;
use std::f64::consts::{PI, TAU};

/// Analytical shader palette family shared across CPU rasterizers and exported WGSL/GLSL shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShaderPalette {
    /// Smooth multi-frequency cosine escape bands (`0`).
    #[default]
    CyclicCosine,
    /// Blue-green-red scientific heatmap ramp (`1`).
    Heatmap,
    /// Five-stop blue-cyan-green-yellow-red gradient (`2`).
    MappedGradient,
    /// Continuous $2\pi$-periodic HSV/triangular complex phase wheel (`3`).
    PhaseWheel,
}

impl ShaderPalette {
    /// Ordered slice of all available analytical shader palettes.
    pub const ALL: [Self; 4] = [
        Self::CyclicCosine,
        Self::Heatmap,
        Self::MappedGradient,
        Self::PhaseWheel,
    ];

    /// Stable numeric uniform identifier (`0..=3`) embedded in exported shaders.
    #[must_use]
    pub const fn palette_id(self) -> u32 {
        match self {
            Self::CyclicCosine => 0,
            Self::Heatmap => 1,
            Self::MappedGradient => 2,
            Self::PhaseWheel => 3,
        }
    }

    /// Compact ASCII identifier used for JSON state and URL bookmark serialization.
    #[must_use]
    pub const fn as_slug(self) -> &'static str {
        match self {
            Self::CyclicCosine => "cosine",
            Self::Heatmap => "heatmap",
            Self::MappedGradient => "gradient",
            Self::PhaseWheel => "phase_wheel",
        }
    }

    /// Resolves a [`ShaderPalette`] from its compact slug (case-insensitive).
    #[must_use]
    pub fn from_slug(slug: &str) -> Option<Self> {
        match slug.trim().to_ascii_lowercase().as_str() {
            "cosine" | "cyclic_cosine" | "cycliccosine" => Some(Self::CyclicCosine),
            "heatmap" => Some(Self::Heatmap),
            "gradient" | "mapped_gradient" | "mappedgradient" => Some(Self::MappedGradient),
            "phase_wheel" | "phasewheel" | "phase" => Some(Self::PhaseWheel),
            _ => None,
        }
    }

    /// Human-readable UI label for combo boxes and preset summaries.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::CyclicCosine => "Cyclic Cosine",
            Self::Heatmap => "Heatmap (Blue-Green-Red)",
            Self::MappedGradient => "Spectrum Gradient",
            Self::PhaseWheel => "Phase Wheel",
        }
    }
}

fn phase_wheel_unit(phase_rad: f64) -> [f64; 3] {
    if !phase_rad.is_finite() {
        return [0.0, 0.0, 0.0];
    }
    let h = (phase_rad / TAU).rem_euclid(1.0);
    let r = (((h * 6.0 - 3.0).abs()) - 1.0).clamp(0.0, 1.0);
    let g = ((((h + 2.0 / 3.0).rem_euclid(1.0) * 6.0 - 3.0).abs()) - 1.0).clamp(0.0, 1.0);
    let b = ((((h + 1.0 / 3.0).rem_euclid(1.0) * 6.0 - 3.0).abs()) - 1.0).clamp(0.0, 1.0);
    [r, g, b]
}

/// Maps a complex phase angle in radians to an RGB color on a continuous $2\pi$-periodic hue wheel.
#[inline]
#[must_use]
pub fn phase_wheel_color(phase_rad: f64) -> [u8; 3] {
    phase_wheel_unit(phase_rad).map(|ch| (ch * 255.0) as u8)
}

/// High-contrast variant of [`phase_wheel_color`] guaranteed to exceed `3.0:1` contrast on `#000000`.
#[inline]
#[must_use]
pub fn phase_wheel_color_high_contrast(phase_rad: f64) -> [u8; 3] {
    ensure_dark_canvas_contrast(phase_wheel_color(phase_rad))
}

/// Evaluates an escape-time scalar `iter_val` in `[0, max_iter]` under `palette` and optional WCAG `3.0:1` contrast lifting.
#[must_use]
pub fn evaluate_fractal_palette(
    iter_val: f64,
    max_iter: u32,
    palette: ShaderPalette,
    high_contrast: bool,
) -> [u8; 3] {
    let denom = f64::from(max_iter.max(1));
    let norm_t = (iter_val / denom).clamp(0.0, 1.0);
    let base = match palette {
        ShaderPalette::CyclicCosine => {
            if high_contrast {
                return cyclic_cosine_palette_high_contrast(iter_val);
            }
            cyclic_cosine_palette(iter_val)
        }
        ShaderPalette::Heatmap => {
            if high_contrast {
                return heatmap_color_high_contrast(norm_t);
            }
            heatmap_color(norm_t)
        }
        ShaderPalette::MappedGradient => mapped_gradient(norm_t, 0.0, 1.0),
        ShaderPalette::PhaseWheel => phase_wheel_color(iter_val * 0.25),
    };
    if high_contrast {
        ensure_dark_canvas_contrast(base)
    } else {
        base
    }
}

/// Computes continuous complex domain coloring (phase hue + octave log-modulus contours) for $w = \text{re} + i\,\text{im}$.
#[must_use]
pub fn domain_color_with_palette(
    re: f64,
    im: f64,
    palette: ShaderPalette,
    high_contrast: bool,
) -> [u8; 3] {
    if !re.is_finite() || !im.is_finite() {
        let fallback = [220, 220, 220];
        return if high_contrast {
            ensure_dark_canvas_contrast(fallback)
        } else {
            fallback
        };
    }
    let phase = im.atan2(re);
    let norm_phase = ((phase + PI) / TAU).clamp(0.0, 1.0);
    let r = re.hypot(im);
    let log_r = r.max(1.0e-12).log2();
    let ring = 0.78 + 0.22 * (0.5 + 0.5 * (TAU * log_r).cos());
    let env = 0.20 + 0.80 * (r / (r + 1.0));
    let shade = (ring * env).clamp(0.0, 1.0);

    let base_u8 = match palette {
        ShaderPalette::PhaseWheel => phase_wheel_color(phase),
        ShaderPalette::CyclicCosine => cyclic_cosine_palette(norm_phase * 40.0 + log_r * 2.0),
        ShaderPalette::Heatmap => heatmap_color(norm_phase),
        ShaderPalette::MappedGradient => mapped_gradient(norm_phase, 0.0, 1.0),
    };
    let shaded = [
        (f64::from(base_u8[0]) * shade) as u8,
        (f64::from(base_u8[1]) * shade) as u8,
        (f64::from(base_u8[2]) * shade) as u8,
    ];
    if high_contrast {
        ensure_dark_canvas_contrast(shaded)
    } else {
        shaded
    }
}

/// Computes phase-wheel continuous domain coloring for $w = \text{re} + i\,\text{im}$.
#[inline]
#[must_use]
pub fn domain_color(re: f64, im: f64, high_contrast: bool) -> [u8; 3] {
    domain_color_with_palette(re, im, ShaderPalette::PhaseWheel, high_contrast)
}

const WGSL_COLORMAP_PRELUDE: &str = r#"fn om_heatmap(t: f32) -> vec3<f32> {
    let tc = clamp(t, 0.0, 1.0);
    let r = max(tc * 2.0 - 1.0, 0.0);
    let g = max(1.0 - abs(tc * 2.0 - 1.0), 0.0);
    let b = max(1.0 - tc * 2.0, 0.0);
    return vec3<f32>(r, g, b);
}

fn om_cosine_palette(n: f32) -> vec3<f32> {
    return vec3<f32>(
        0.5 + 0.5 * cos(3.0 + n * 0.15),
        0.5 + 0.5 * cos(5.0 + n * 0.15),
        0.5 + 0.5 * cos(7.0 + n * 0.15)
    );
}

fn om_mapped_gradient(value: f32, min_val: f32, max_val: f32) -> vec3<f32> {
    let span = max_val - min_val;
    let n = select(clamp((value - min_val) / span, 0.0, 1.0), 0.5, abs(span) < 1.0e-12);
    let r = clamp((n - 0.5) * 4.0, 0.0, 1.0);
    let g = min(clamp(n * 4.0, 0.0, 1.0), clamp((1.0 - n) * 4.0, 0.0, 1.0));
    let b = clamp((0.5 - n) * 4.0, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn om_phase_wheel(phase_rad: f32) -> vec3<f32> {
    let h = fract(phase_rad * 0.15915494);
    let r = clamp(abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    let g = clamp(abs(fract(h + 0.6666667) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    let b = clamp(abs(fract(h + 0.33333334) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

fn om_srgb_to_linear_ch(s: f32) -> f32 {
    let c = clamp(s, 0.0, 1.0);
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn om_linear_to_srgb_ceil_ch(lin: f32) -> f32 {
    let c = clamp(lin, 0.0, 1.0);
    let enc = select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
    return ceil(clamp(enc, 0.0, 1.0) * 255.0) / 255.0;
}

fn om_ensure_contrast(rgb: vec3<f32>) -> vec3<f32> {
    let lr = om_srgb_to_linear_ch(rgb.x);
    let lg = om_srgb_to_linear_ch(rgb.y);
    let lb = om_srgb_to_linear_ch(rgb.z);
    let lum = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
    if (lum >= 0.11) {
        return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let alpha = clamp((0.11 - lum) / (1.0 - lum), 0.0, 1.0);
    return vec3<f32>(
        om_linear_to_srgb_ceil_ch(lr + alpha * (1.0 - lr)),
        om_linear_to_srgb_ceil_ch(lg + alpha * (1.0 - lg)),
        om_linear_to_srgb_ceil_ch(lb + alpha * (1.0 - lb))
    );
}

fn om_eval_palette(iter_val: f32, max_iter: u32, palette_id: u32, high_contrast: u32) -> vec3<f32> {
    let norm_t = clamp(iter_val / f32(max(max_iter, 1u)), 0.0, 1.0);
    var col = om_cosine_palette(iter_val);
    if (palette_id == 1u) {
        col = om_heatmap(norm_t);
    } else if (palette_id == 2u) {
        col = om_mapped_gradient(norm_t, 0.0, 1.0);
    } else if (palette_id == 3u) {
        col = om_phase_wheel(iter_val * 0.25);
    }
    if (high_contrast != 0u) {
        col = om_ensure_contrast(col);
    }
    return col;
}

fn om_domain_color(w: vec2<f32>, palette_id: u32, high_contrast: u32) -> vec3<f32> {
    let phase = atan2(w.y, w.x);
    let norm_phase = clamp((phase + 3.1415927) * 0.15915494, 0.0, 1.0);
    let r = length(w);
    let log_r = log2(max(r, 1.0e-12));
    let ring = 0.78 + 0.22 * (0.5 + 0.5 * cos(6.2831855 * log_r));
    let env = 0.20 + 0.80 * (r / (r + 1.0));
    let shade = clamp(ring * env, 0.0, 1.0);
    var base = om_phase_wheel(phase);
    if (palette_id == 0u) {
        base = om_cosine_palette(norm_phase * 40.0 + log_r * 2.0);
    } else if (palette_id == 1u) {
        base = om_heatmap(norm_phase);
    } else if (palette_id == 2u) {
        base = om_mapped_gradient(norm_phase, 0.0, 1.0);
    }
    var col = base * shade;
    if (high_contrast != 0u) {
        col = om_ensure_contrast(col);
    }
    return col;
}
"#;

/// Emits analytical WGSL functions for heatmaps, cosine palettes, spectrum gradients, phase wheels,
/// domain coloring, and WCAG 2.2 `3.0:1` linear-light contrast lifting.
#[must_use]
pub const fn wgsl_colormap_prelude() -> &'static str {
    WGSL_COLORMAP_PRELUDE
}

const GLSL_COLORMAP_PRELUDE: &str = r#"vec3 om_heatmap(float t) {
    float tc = clamp(t, 0.0, 1.0);
    float r = max(tc * 2.0 - 1.0, 0.0);
    float g = max(1.0 - abs(tc * 2.0 - 1.0), 0.0);
    float b = max(1.0 - tc * 2.0, 0.0);
    return vec3(r, g, b);
}

vec3 om_cosine_palette(float n) {
    return vec3(
        0.5 + 0.5 * cos(3.0 + n * 0.15),
        0.5 + 0.5 * cos(5.0 + n * 0.15),
        0.5 + 0.5 * cos(7.0 + n * 0.15)
    );
}

vec3 om_mapped_gradient(float value, float min_val, float max_val) {
    float span = max_val - min_val;
    float n = (abs(span) < 1.0e-12) ? 0.5 : clamp((value - min_val) / span, 0.0, 1.0);
    float r = clamp((n - 0.5) * 4.0, 0.0, 1.0);
    float g = min(clamp(n * 4.0, 0.0, 1.0), clamp((1.0 - n) * 4.0, 0.0, 1.0));
    float b = clamp((0.5 - n) * 4.0, 0.0, 1.0);
    return vec3(r, g, b);
}

vec3 om_phase_wheel(float phase_rad) {
    float h = fract(phase_rad * 0.15915494);
    float r = clamp(abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    float g = clamp(abs(fract(h + 0.6666667) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    float b = clamp(abs(fract(h + 0.33333334) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return vec3(r, g, b);
}

float om_srgb_to_linear_ch(float s) {
    float c = clamp(s, 0.0, 1.0);
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

float om_linear_to_srgb_ceil_ch(float lin) {
    float c = clamp(lin, 0.0, 1.0);
    float enc = (c <= 0.0031308) ? (c * 12.92) : (1.055 * pow(c, 1.0 / 2.4) - 0.055);
    return ceil(clamp(enc, 0.0, 1.0) * 255.0) / 255.0;
}

vec3 om_ensure_contrast(vec3 rgb) {
    float lr = om_srgb_to_linear_ch(rgb.x);
    float lg = om_srgb_to_linear_ch(rgb.y);
    float lb = om_srgb_to_linear_ch(rgb.z);
    float lum = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
    if (lum >= 0.11) {
        return clamp(rgb, vec3(0.0), vec3(1.0));
    }
    float alpha = clamp((0.11 - lum) / (1.0 - lum), 0.0, 1.0);
    return vec3(
        om_linear_to_srgb_ceil_ch(lr + alpha * (1.0 - lr)),
        om_linear_to_srgb_ceil_ch(lg + alpha * (1.0 - lg)),
        om_linear_to_srgb_ceil_ch(lb + alpha * (1.0 - lb))
    );
}

vec3 om_eval_palette(float iter_val, int max_iter, int palette_id, int high_contrast) {
    float norm_t = clamp(iter_val / float(max(max_iter, 1)), 0.0, 1.0);
    vec3 col = om_cosine_palette(iter_val);
    if (palette_id == 1) {
        col = om_heatmap(norm_t);
    } else if (palette_id == 2) {
        col = om_mapped_gradient(norm_t, 0.0, 1.0);
    } else if (palette_id == 3) {
        col = om_phase_wheel(iter_val * 0.25);
    }
    if (high_contrast != 0) {
        col = om_ensure_contrast(col);
    }
    return col;
}

vec3 om_domain_color(vec2 w, int palette_id, int high_contrast) {
    float phase = atan(w.y, w.x);
    float norm_phase = clamp((phase + 3.1415927) * 0.15915494, 0.0, 1.0);
    float r = length(w);
    float log_r = log2(max(r, 1.0e-12));
    float ring = 0.78 + 0.22 * (0.5 + 0.5 * cos(6.2831855 * log_r));
    float env = 0.20 + 0.80 * (r / (r + 1.0));
    float shade = clamp(ring * env, 0.0, 1.0);
    vec3 base = om_phase_wheel(phase);
    if (palette_id == 0) {
        base = om_cosine_palette(norm_phase * 40.0 + log_r * 2.0);
    } else if (palette_id == 1) {
        base = om_heatmap(norm_phase);
    } else if (palette_id == 2) {
        base = om_mapped_gradient(norm_phase, 0.0, 1.0);
    }
    vec3 col = base * shade;
    if (high_contrast != 0) {
        col = om_ensure_contrast(col);
    }
    return col;
}
"#;

/// Emits analytical GLSL ES 3.00 functions for heatmaps, cosine palettes, spectrum gradients,
/// phase wheels, domain coloring, and WCAG 2.2 `3.0:1` linear-light contrast lifting.
#[must_use]
pub const fn glsl_colormap_prelude() -> &'static str {
    GLSL_COLORMAP_PRELUDE
}

/// Returns the analytical colormap prelude for `target`.
#[must_use]
pub const fn colormap_prelude_for(target: ShaderTarget) -> &'static str {
    match target {
        ShaderTarget::Wgsl => wgsl_colormap_prelude(),
        ShaderTarget::Glsl => glsl_colormap_prelude(),
    }
}
