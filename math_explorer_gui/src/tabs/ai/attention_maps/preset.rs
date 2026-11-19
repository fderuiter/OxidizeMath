#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::AttentionMapsTool;
use nalgebra::DMatrix;

pub(super) fn render_attention_raster(
    weights: &DMatrix<f64>,
    output: &DMatrix<f64>,
) -> Option<eframe::egui::ColorImage> {
    if weights.nrows() == 0 || weights.ncols() == 0 {
        return None;
    }
    let gap = 1usize;
    let total_cols = weights.ncols() + gap + output.ncols();
    let total_rows = weights.nrows().max(output.nrows());
    let mut pixels = vec![eframe::egui::Color32::from_gray(24); total_cols * total_rows];

    let paint_mat = |mat: &DMatrix<f64>, col_offset: usize, buf: &mut [eframe::egui::Color32]| {
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AttentionMapsState {
    rows: usize,
    cols: usize,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
}

fn flatten_row_major(matrix: &DMatrix<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for r in 0..matrix.nrows() {
        for c in 0..matrix.ncols() {
            out.push(matrix[(r, c)]);
        }
    }
    out
}

impl AttentionMapsTool {
    pub(super) fn save_preset_state(&self) -> Option<String> {
        serde_json::to_string(&AttentionMapsState {
            rows: self.q_matrix.nrows(),
            cols: self.q_matrix.ncols(),
            q: flatten_row_major(&self.q_matrix),
            k: flatten_row_major(&self.k_matrix),
            v: flatten_row_major(&self.v_matrix),
        })
        .ok()
    }

    pub(super) fn try_load_preset_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: AttentionMapsState =
            serde_json::from_str(state).map_err(|e| format!("Invalid attention state: {e}"))?;
        let expected_len = parsed.rows.saturating_mul(parsed.cols);
        if !(1..=32).contains(&parsed.rows)
            || !(1..=32).contains(&parsed.cols)
            || parsed.q.len() != expected_len
            || parsed.k.len() != expected_len
            || parsed.v.len() != expected_len
            || !parsed
                .q
                .iter()
                .chain(&parsed.k)
                .chain(&parsed.v)
                .all(|x| x.is_finite() && (-10.0..=10.0).contains(x))
        {
            return Err("Attention matrices must have matching finite bounded dimensions".into());
        }
        self.q_matrix = DMatrix::from_row_slice(parsed.rows, parsed.cols, &parsed.q);
        self.k_matrix = DMatrix::from_row_slice(parsed.rows, parsed.cols, &parsed.k);
        self.v_matrix = DMatrix::from_row_slice(parsed.rows, parsed.cols, &parsed.v);
        self.recalculate();
        Ok(())
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Self-Attention Heatmap",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "rows": 4,
    "cols": 4,
    "q": [1.2, 0.1, -0.3, 0.4, 0.2, 1.5, 0.1, -0.2, -0.4, 0.3, 1.4, 0.2, 0.5, -0.1, 0.2, 1.3],
    "k": [1.1, 0.2, -0.2, 0.3, 0.1, 1.4, 0.2, -0.3, -0.3, 0.2, 1.5, 0.1, 0.4, -0.2, 0.1, 1.2],
    "v": [1.0, 0.0, 0.5, -0.5, 0.0, 1.0, -0.5, 0.5, 0.5, -0.5, 1.0, 0.0, -0.5, 0.5, 0.0, 1.0]
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "attention_maps_starter",
        title: "Self-Attention Heatmap",
        description: "Softmax(QKᵀ / √d_k)V: Scaled dot-product query-key affinity and value projection heatmap.",
        category: "Transformers",
        icon: "🔍",
        domain: "ai",
        tool_name: Some("AttentionMapsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
