//! GLSL ES 3.00 colormap and WCAG 2.2 dark-canvas contrast emitters mirroring [`super::colormap`].
// [cite:math_commons]

/// Emits the GLSL ES 3.00 `heatmap_color(float t)` function mapping `[0.0, 1.0]` to RGB (`vec3`).
#[must_use]
pub const fn glsl_heatmap_color_fn() -> &'static str {
    r#"vec3 heatmap_color(float t) {
    float tc = clamp(t, 0.0, 1.0);
    float r = max(tc * 2.0 - 1.0, 0.0);
    float g = max(1.0 - abs(tc * 2.0 - 1.0), 0.0);
    float b = max(1.0 - tc * 2.0, 0.0);
    return vec3(r, g, b);
}
"#
}

/// Emits the GLSL ES 3.00 `cyclic_cosine_palette(float n)` function for smooth escape-time and phase coloring.
#[must_use]
pub const fn glsl_cyclic_cosine_palette_fn() -> &'static str {
    r#"vec3 cyclic_cosine_palette(float n) {
    return vec3(
        0.5 + 0.5 * cos(3.0 + n * 0.15),
        0.5 + 0.5 * cos(5.0 + n * 0.15),
        0.5 + 0.5 * cos(7.0 + n * 0.15)
    );
}
"#
}

/// Emits the GLSL ES 3.00 WCAG 2.2 `>= 3.0:1` linear-light `ensure_dark_canvas_contrast(vec3 srgb)` guard.
#[must_use]
pub const fn glsl_ensure_dark_canvas_contrast_fn() -> &'static str {
    r#"float srgb_to_linear_channel(float s) {
    float sc = clamp(s, 0.0, 1.0);
    return (sc <= 0.04045) ? (sc / 12.92) : pow((sc + 0.055) / 1.055, 2.4);
}

float linear_to_srgb_channel(float linear) {
    float lc = clamp(linear, 0.0, 1.0);
    float encoded = (lc <= 0.0031308) ? (lc * 12.92) : (1.055 * pow(lc, 1.0 / 2.4) - 0.055);
    return clamp(ceil(encoded * 255.0) / 255.0, 0.0, 1.0);
}

vec3 ensure_dark_canvas_contrast(vec3 srgb) {
    vec3 lin = vec3(
        srgb_to_linear_channel(srgb.r),
        srgb_to_linear_channel(srgb.g),
        srgb_to_linear_channel(srgb.b)
    );
    float lum = dot(lin, vec3(0.2126, 0.7152, 0.0722));
    if (lum >= 0.11) {
        return clamp(srgb, 0.0, 1.0);
    }
    float alpha = clamp((0.11 - lum) / (1.0 - lum), 0.0, 1.0);
    vec3 lifted = lin + alpha * (vec3(1.0) - lin);
    return vec3(
        linear_to_srgb_channel(lifted.r),
        linear_to_srgb_channel(lifted.g),
        linear_to_srgb_channel(lifted.b)
    );
}
"#
}

/// Returns a concatenated GLSL ES 3.00 colormap library containing `heatmap_color`,
/// `cyclic_cosine_palette`, and WCAG 2.2 `ensure_dark_canvas_contrast`.
#[must_use]
pub fn glsl_colormap_library() -> String {
    let mut out = String::with_capacity(1024);
    out.push_str(glsl_heatmap_color_fn());
    out.push('\n');
    out.push_str(glsl_cyclic_cosine_palette_fn());
    out.push('\n');
    out.push_str(glsl_ensure_dark_canvas_contrast_fn());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_kernel::colormap::{
        cyclic_cosine_palette, ensure_dark_canvas_contrast, heatmap_color,
    };

    fn sim_srgb_to_linear(s: f32) -> f32 {
        let sc = s.clamp(0.0, 1.0);
        if sc <= 0.04045 {
            sc / 12.92
        } else {
            ((sc + 0.055) / 1.055).powf(2.4)
        }
    }

    fn sim_linear_to_srgb(linear: f32) -> f32 {
        let lc = linear.clamp(0.0, 1.0);
        let encoded = if lc <= 0.003_130_8 {
            lc * 12.92
        } else {
            1.055 * lc.powf(1.0 / 2.4) - 0.055
        };
        ((encoded * 255.0).ceil() / 255.0).clamp(0.0, 1.0)
    }

    fn sim_ensure_dark_canvas_contrast(srgb: [f32; 3]) -> [f32; 3] {
        let lin = [
            sim_srgb_to_linear(srgb[0]),
            sim_srgb_to_linear(srgb[1]),
            sim_srgb_to_linear(srgb[2]),
        ];
        let lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
        if lum >= 0.11 {
            return srgb.map(|c| c.clamp(0.0, 1.0));
        }
        let alpha = ((0.11 - lum) / (1.0 - lum)).clamp(0.0, 1.0);
        [
            sim_linear_to_srgb(lin[0] + alpha * (1.0 - lin[0])),
            sim_linear_to_srgb(lin[1] + alpha * (1.0 - lin[1])),
            sim_linear_to_srgb(lin[2] + alpha * (1.0 - lin[2])),
        ]
    }

    #[test]
    fn test_glsl_colormap_emitters_contain_all_signatures() {
        let lib = glsl_colormap_library();
        assert!(lib.contains("vec3 heatmap_color(float t)"));
        assert!(lib.contains("vec3 cyclic_cosine_palette(float n)"));
        assert!(lib.contains("vec3 ensure_dark_canvas_contrast(vec3 srgb)"));
        assert!(lib.contains("0.2126") && lib.contains("0.7152") && lib.contains("0.0722"));
    }

    #[test]
    fn test_glsl_colormap_parity_and_wcag_3_to_1_contrast_on_black() {
        for step in 0..=200 {
            let t = step as f32 / 200.0;
            let cpu = heatmap_color(f64::from(t));
            let gpu = [
                (t * 2.0 - 1.0).max(0.0),
                (1.0 - (t * 2.0 - 1.0).abs()).max(0.0),
                (1.0 - t * 2.0).max(0.0),
            ];
            for ch in 0..3 {
                assert!(((gpu[ch] * 255.0) as i32 - i32::from(cpu[ch])).abs() <= 1);
            }
            let lifted = sim_ensure_dark_canvas_contrast(gpu);
            let lum = 0.2126 * sim_srgb_to_linear(lifted[0])
                + 0.7152 * sim_srgb_to_linear(lifted[1])
                + 0.0722 * sim_srgb_to_linear(lifted[2]);
            let ratio = (lum + 0.05) / 0.05;
            assert!(ratio >= 3.0, "Contrast {ratio:.2}:1 < 3.0 at t={t}");
        }

        for iter in 0..=500 {
            let n = iter as f32;
            let cpu = ensure_dark_canvas_contrast(cyclic_cosine_palette(f64::from(n)));
            let gpu_raw = [
                0.5 + 0.5 * (3.0 + n * 0.15).cos(),
                0.5 + 0.5 * (5.0 + n * 0.15).cos(),
                0.5 + 0.5 * (7.0 + n * 0.15).cos(),
            ];
            let gpu_hc = sim_ensure_dark_canvas_contrast(gpu_raw);
            let lum = 0.2126 * sim_srgb_to_linear(gpu_hc[0])
                + 0.7152 * sim_srgb_to_linear(gpu_hc[1])
                + 0.0722 * sim_srgb_to_linear(gpu_hc[2]);
            let ratio = (lum + 0.05) / 0.05;
            assert!(ratio >= 3.0, "Cyclic contrast {ratio:.2}:1 < 3.0 at n={n}");
            for ch in 0..3 {
                let gpu_u8 = (gpu_hc[ch] * 255.0).floor() as i32;
                assert!(
                    (gpu_u8 - i32::from(cpu[ch])).abs() <= 4,
                    "n={n} ch={ch}: gpu={gpu_u8}, cpu={}",
                    cpu[ch]
                );
            }
        }
    }
}
