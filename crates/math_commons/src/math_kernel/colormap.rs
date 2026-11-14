//! A unified colormap utility module for scalar-to-color mapping.
//! Handles custom color-gradient and step-linear mappings across the codebase.

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
