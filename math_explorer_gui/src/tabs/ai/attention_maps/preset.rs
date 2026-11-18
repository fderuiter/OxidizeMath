#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub(super) fn render_attention_raster(
    weights: &nalgebra::DMatrix<f64>,
    output: &nalgebra::DMatrix<f64>,
) -> Option<eframe::egui::ColorImage> {
    if weights.nrows() == 0 || weights.ncols() == 0 {
        return None;
    }
    let gap = 1usize;
    let total_cols = weights.ncols() + gap + output.ncols();
    let total_rows = weights.nrows().max(output.nrows());
    let mut pixels = vec![eframe::egui::Color32::from_gray(24); total_cols * total_rows];

    let paint_mat =
        |mat: &nalgebra::DMatrix<f64>, col_offset: usize, buf: &mut [eframe::egui::Color32]| {
            let (min_v, max_v) = mat
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(v), hi.max(v))
                });
            let (min_v, max_v) = if min_v.is_finite() && max_v.is_finite() {
                (min_v, max_v)
            } else {
                (0.0, 1.0)
            };
            for r in 0..mat.nrows() {
                for c in 0..mat.ncols() {
                    let norm = if (max_v - min_v).abs() > 1e-6 {
                        (mat[(r, c)] - min_v) / (max_v - min_v)
                    } else {
                        0.5
                    };
                    buf[r * total_cols + col_offset + c] =
                        super::compute_attention_cell_color(norm, super::ThemeContrastMode::Dark);
                }
            }
        };

    paint_mat(weights, 0, &mut pixels);
    if output.nrows() > 0 && output.ncols() > 0 {
        paint_mat(output, weights.ncols() + gap, &mut pixels);
    }
    Some(eframe::egui::ColorImage::new(
        [total_cols, total_rows],
        pixels,
    ))
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Self-Attention Heatmap",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "num_heads": 8,
    "sequence_length": 16
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "attention_maps_starter",
        title: "Self-Attention Heatmap",
        description: "Softmax(QKᵀ / √d_k)V: Multi-head transformer self-attention query-key affinity visualization.",
        category: "Transformers",
        icon: "🔍",
        domain: "ai",
        tool_name: Some("AttentionMapsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
