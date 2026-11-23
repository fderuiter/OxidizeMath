//! A unified colormap utility module for scalar-to-color mapping.
//! Handles custom color-gradient and step-linear mappings across the codebase.
// [cite:math_commons]

/// Maps a normalized scalar value (0.0 to 1.0) to an RGB color based on a typical heatmap (blue -> green -> red).
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::heatmap_color;
///
/// assert_eq!(heatmap_color(0.0), [0, 0, 255]);
/// assert_eq!(heatmap_color(1.0), [255, 0, 0]);
/// ```
#[inline(always)]
pub fn heatmap_color(mut t: f64) -> [u8; 3] {
    t = t.clamp(0.0, 1.0);

    // Low: Blue, Mid: Green, High: Red
    let r = (t * 2.0 - 1.0).max(0.0);
    let g = (1.0 - (t * 2.0 - 1.0).abs()).max(0.0);
    let b = (1.0 - t * 2.0).max(0.0);

    let r_u8 = (r * 255.0) as u8;
    let g_u8 = (g * 255.0) as u8;
    let b_u8 = (b * 255.0) as u8;

    [r_u8, g_u8, b_u8]
}

/// Maps a cyclic value to a color (e.g. for escape-time fractals).
#[inline(always)]
#[allow(missing_docs)]
pub fn cyclic_cosine_palette(n: f64) -> [u8; 3] {
    let r = (0.5 + 0.5 * (3.0 + n * 0.15).cos()) * 255.0;
    let g = (0.5 + 0.5 * (3.0 + n * 0.15 + 2.0).cos()) * 255.0;
    let b = (0.5 + 0.5 * (3.0 + n * 0.15 + 4.0).cos()) * 255.0;
    [r as u8, g as u8, b as u8]
}

/// Maps a normalized scalar value using a step-linear gradient.
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::mapped_gradient;
///
/// let blue = mapped_gradient(0.0, 0.0, 100.0);
/// assert_eq!(blue, [0, 0, 255]);
/// let red = mapped_gradient(100.0, 0.0, 100.0);
/// assert_eq!(red, [255, 0, 0]);
/// ```
#[inline(always)]
pub fn mapped_gradient(value: f64, min_val: f64, max_val: f64) -> [u8; 3] {
    let range = max_val - min_val;
    let normalized = if range == 0.0 {
        0.5
    } else {
        ((value - min_val) / range).clamp(0.0, 1.0)
    };

    // Blue -> Cyan -> Green -> Yellow -> Red
    let r = if normalized < 0.5 {
        0
    } else if normalized < 0.75 {
        ((normalized - 0.5) * 4.0 * 255.0) as u8
    } else {
        255
    };

    let g = if normalized < 0.25 {
        (normalized * 4.0 * 255.0) as u8
    } else if normalized < 0.75 {
        255
    } else {
        ((1.0 - normalized) * 4.0 * 255.0) as u8
    };

    let b = if normalized < 0.25 {
        255
    } else if normalized < 0.5 {
        ((0.5 - normalized) * 4.0 * 255.0) as u8
    } else {
        0
    };

    [r, g, b]
}

const MIN_DARK_CANVAS_LUMINANCE: f64 = 0.11;

#[inline]
fn srgb_channel_to_linear(channel: u8) -> f64 {
    let s = f64::from(channel) / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_channel_to_srgb_ceil(linear: f64) -> u8 {
    let clamped = linear.clamp(0.0, 1.0);
    let encoded = if clamped <= 0.003_130_8 {
        clamped * 12.92
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).ceil().clamp(0.0, 255.0) as u8
}

/// Lifts dark RGB samples in linear light so every pixel achieves WCAG 2.2 `>= 3.0:1`
/// graphical contrast against a pure `#000000` canvas while preserving hue ordering.
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::{ensure_dark_canvas_contrast, heatmap_color};
///
/// let lifted = ensure_dark_canvas_contrast(heatmap_color(0.0));
/// assert!(lifted[0] > 0 && lifted[1] > 0 && lifted[2] == 255);
/// ```
#[inline]
#[must_use]
pub fn ensure_dark_canvas_contrast(rgb: [u8; 3]) -> [u8; 3] {
    let lr = srgb_channel_to_linear(rgb[0]);
    let lg = srgb_channel_to_linear(rgb[1]);
    let lb = srgb_channel_to_linear(rgb[2]);
    let lum = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
    if lum >= MIN_DARK_CANVAS_LUMINANCE {
        return rgb;
    }
    let alpha = ((MIN_DARK_CANVAS_LUMINANCE - lum) / (1.0 - lum)).clamp(0.0, 1.0);
    [
        linear_channel_to_srgb_ceil(lr + alpha * (1.0 - lr)),
        linear_channel_to_srgb_ceil(lg + alpha * (1.0 - lg)),
        linear_channel_to_srgb_ceil(lb + alpha * (1.0 - lb)),
    ]
}

/// High-contrast variant of [`heatmap_color`] guaranteed to exceed `3.0:1` contrast on `#000000`.
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::heatmap_color_high_contrast;
///
/// let low = heatmap_color_high_contrast(0.0);
/// assert_eq!(low[2], 255);
/// assert_eq!(heatmap_color_high_contrast(1.0), [255, 0, 0]);
/// ```
#[inline]
#[must_use]
pub fn heatmap_color_high_contrast(t: f64) -> [u8; 3] {
    ensure_dark_canvas_contrast(heatmap_color(t))
}

/// High-contrast variant of [`cyclic_cosine_palette`] guaranteed to exceed `3.0:1` contrast on `#000000`.
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::cyclic_cosine_palette_high_contrast;
///
/// let sample = cyclic_cosine_palette_high_contrast(0.0);
/// assert!(sample.iter().any(|&c| c > 0));
/// ```
#[inline]
#[must_use]
pub fn cyclic_cosine_palette_high_contrast(n: f64) -> [u8; 3] {
    ensure_dark_canvas_contrast(cyclic_cosine_palette(n))
}

#[path = "colormap_shader.rs"]
pub mod colormap_shader;
pub use colormap_shader::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn contrast_on_black(rgb: [u8; 3]) -> f64 {
        let lum = 0.2126 * srgb_channel_to_linear(rgb[0])
            + 0.7152 * srgb_channel_to_linear(rgb[1])
            + 0.0722 * srgb_channel_to_linear(rgb[2]);
        (lum + 0.05) / 0.05
    }

    fn sim_shader_ensure_contrast(rgb: [f32; 3]) -> [u8; 3] {
        let to_lin = |s: f32| {
            let c = s.clamp(0.0, 1.0);
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        let to_srgb_ceil = |lin: f32| {
            let c = lin.clamp(0.0, 1.0);
            let enc = if c <= 0.003_130_8 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            (enc.clamp(0.0, 1.0) * 255.0).ceil() as u8
        };
        let (lr, lg, lb) = (to_lin(rgb[0]), to_lin(rgb[1]), to_lin(rgb[2]));
        let lum = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
        if lum >= 0.11 {
            return [
                (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
            ];
        }
        let alpha = ((0.11 - lum) / (1.0 - lum)).clamp(0.0, 1.0);
        [
            to_srgb_ceil(lr + alpha * (1.0 - lr)),
            to_srgb_ceil(lg + alpha * (1.0 - lg)),
            to_srgb_ceil(lb + alpha * (1.0 - lb)),
        ]
    }

    #[test]
    fn test_high_contrast_colormaps_exceed_wcag_3_to_1_on_black() {
        assert_eq!(heatmap_color(0.0), [0, 0, 255]);
        assert!(contrast_on_black(heatmap_color(0.0)) < 3.0);

        for step in 0..=200 {
            let t = f64::from(step) / 200.0;
            let hc = heatmap_color_high_contrast(t);
            let ratio = contrast_on_black(hc);
            assert!(
                ratio >= 3.0,
                "heatmap_color_high_contrast({t:.3}) -> {hc:?} yielded {ratio:.2}:1 (< 3.0:1)"
            );
        }

        for iter in 0..=500 {
            let hc = cyclic_cosine_palette_high_contrast(f64::from(iter));
            let ratio = contrast_on_black(hc);
            assert!(
                ratio >= 3.0,
                "cyclic_cosine_palette_high_contrast({iter}) -> {hc:?} yielded {ratio:.2}:1 (< 3.0:1)"
            );
        }
    }

    #[test]
    fn test_shader_palettes_match_cpu_and_preserve_high_contrast() {
        let wgsl = wgsl_colormap_prelude();
        let glsl = glsl_colormap_prelude();
        for symbol in [
            "om_heatmap",
            "om_cosine_palette",
            "om_mapped_gradient",
            "om_phase_wheel",
            "om_ensure_contrast",
            "om_eval_palette",
            "om_domain_color",
        ] {
            assert!(wgsl.contains(symbol) && glsl.contains(symbol));
        }

        for step in 0..=200 {
            let t = f64::from(step) / 200.0;
            let tf = t as f32;
            let cpu_hm = heatmap_color(t);
            let gpu_hm = [
                ((tf * 2.0 - 1.0).max(0.0) * 255.0) as u8,
                ((1.0 - (tf * 2.0 - 1.0).abs()).max(0.0) * 255.0) as u8,
                ((1.0 - tf * 2.0).max(0.0) * 255.0) as u8,
            ];
            for ch in 0..3 {
                assert!(cpu_hm[ch].abs_diff(gpu_hm[ch]) <= 1);
            }
            let lifted = sim_shader_ensure_contrast([
                (tf * 2.0 - 1.0).max(0.0),
                (1.0 - (tf * 2.0 - 1.0).abs()).max(0.0),
                (1.0 - tf * 2.0).max(0.0),
            ]);
            assert!(contrast_on_black(lifted) >= 3.0);
        }

        for palette in ShaderPalette::ALL {
            assert_eq!(ShaderPalette::from_slug(palette.as_slug()), Some(palette));
            let dom_hc = domain_color_with_palette(0.0, 0.0, palette, true);
            assert!(contrast_on_black(dom_hc) >= 3.0);
            let frac_hc = evaluate_fractal_palette(3.7, 100, palette, true);
            assert!(contrast_on_black(frac_hc) >= 3.0);
        }
        assert_eq!(phase_wheel_color(0.0), [255, 0, 0]);
        assert!(contrast_on_black(phase_wheel_color_high_contrast(-2.1)) >= 3.0);
    }
}
