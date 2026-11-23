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

/// Maps normalized signed vorticity `omega_norm` in `[-1.0, 1.0]` to a diverging dark-canvas palette
/// (cyan-blue clockwise `[-1]` -> slate `[0]` -> coral-amber counter-clockwise `[+1]`).
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::vorticity_color;
///
/// let neg = vorticity_color(-1.0);
/// let pos = vorticity_color(1.0);
/// assert!(neg[2] > neg[0]);
/// assert!(pos[0] > pos[2]);
/// ```
#[inline]
#[must_use]
pub fn vorticity_color(omega_norm: f64) -> [u8; 3] {
    let w = if omega_norm.is_finite() {
        omega_norm.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    if w < 0.0 {
        let t = -w;
        let r = (24.0 * (1.0 - t)) as u8;
        let g = (30.0 + 158.0 * t) as u8;
        let b = (44.0 + 211.0 * t) as u8;
        [r, g, b]
    } else {
        let t = w;
        let r = (24.0 + 231.0 * t) as u8;
        let g = (30.0 + 78.0 * t) as u8;
        let b = (44.0 * (1.0 - 0.5 * t)) as u8;
        [r, g, b]
    }
}

/// Maps normalized signed vorticity `omega_norm` in `[-1.0, 1.0]` to a crisp light-theme diverging palette
/// (cobalt `[-1]` -> alabaster `[0]` -> vermilion `[+1]`).
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::vorticity_color_light;
///
/// let mid = vorticity_color_light(0.0);
/// assert!(mid[0] > 230 && mid[1] > 230 && mid[2] > 230);
/// ```
#[inline]
#[must_use]
pub fn vorticity_color_light(omega_norm: f64) -> [u8; 3] {
    let w = if omega_norm.is_finite() {
        omega_norm.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    if w < 0.0 {
        let t = -w;
        let r = (242.0 * (1.0 - t) + 18.0 * t) as u8;
        let g = (245.0 * (1.0 - t) + 98.0 * t) as u8;
        let b = (249.0 * (1.0 - t) + 212.0 * t) as u8;
        [r, g, b]
    } else {
        let t = w;
        let r = (242.0 * (1.0 - t) + 216.0 * t) as u8;
        let g = (245.0 * (1.0 - t) + 56.0 * t) as u8;
        let b = (249.0 * (1.0 - t) + 20.0 * t) as u8;
        [r, g, b]
    }
}

/// High-contrast variant of [`vorticity_color`] guaranteed to exceed `3.0:1` contrast on `#000000`.
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::vorticity_color_high_contrast;
///
/// let zero = vorticity_color_high_contrast(0.0);
/// assert!(zero.iter().all(|&c| c > 70));
/// ```
#[inline]
#[must_use]
pub fn vorticity_color_high_contrast(omega_norm: f64) -> [u8; 3] {
    ensure_dark_canvas_contrast(vorticity_color(omega_norm))
}

/// Composites normalized reagent concentration (`[0.0, 1.0]`) with signed vorticity (`[-1.0, 1.0]`)
/// for dual morphogenesis + fluid shear visualization across Dark, Light, and HighContrast themes.
///
/// # Example
/// ```
/// use math_commons::math_kernel::colormap::dual_concentration_vorticity_color;
///
/// let sample = dual_concentration_vorticity_color(0.5, -0.8, false, true);
/// assert!(sample.iter().any(|&c| c > 100));
/// ```
#[inline]
#[must_use]
pub fn dual_concentration_vorticity_color(
    conc_norm: f64,
    omega_norm: f64,
    light_theme: bool,
    high_contrast: bool,
) -> [u8; 3] {
    let c = if conc_norm.is_finite() {
        conc_norm.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let w = if omega_norm.is_finite() {
        omega_norm.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let base = heatmap_color(c);
    let vort = if light_theme && !high_contrast {
        vorticity_color_light(w)
    } else {
        vorticity_color(w)
    };
    let blend = (w.abs() * 0.65).clamp(0.0, 0.65);
    let keep = 1.0 - blend;
    let rgb = [
        (f64::from(base[0]) * keep + f64::from(vort[0]) * blend) as u8,
        (f64::from(base[1]) * keep + f64::from(vort[1]) * blend) as u8,
        (f64::from(base[2]) * keep + f64::from(vort[2]) * blend) as u8,
    ];
    if high_contrast {
        ensure_dark_canvas_contrast(rgb)
    } else {
        rgb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contrast_on_black(rgb: [u8; 3]) -> f64 {
        let lum = 0.2126 * srgb_channel_to_linear(rgb[0])
            + 0.7152 * srgb_channel_to_linear(rgb[1])
            + 0.0722 * srgb_channel_to_linear(rgb[2]);
        (lum + 0.05) / 0.05
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

            let omega = t * 2.0 - 1.0;
            let vort_hc = vorticity_color_high_contrast(omega);
            let vort_ratio = contrast_on_black(vort_hc);
            assert!(
                vort_ratio >= 3.0,
                "vorticity_color_high_contrast({omega:.3}) -> {vort_hc:?} yielded {vort_ratio:.2}:1"
            );

            let dual_hc = dual_concentration_vorticity_color(t, -omega, false, true);
            let dual_ratio = contrast_on_black(dual_hc);
            assert!(
                dual_ratio >= 3.0,
                "dual_concentration_vorticity_color({t:.3}, {}) -> {dual_hc:?} yielded {dual_ratio:.2}:1",
                -omega
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
}
