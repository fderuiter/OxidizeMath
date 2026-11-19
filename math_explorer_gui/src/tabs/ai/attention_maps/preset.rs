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
struct AttentionMapsState {
    seq_len: usize,
    d_k: usize,
    q_values: Vec<f64>,
    k_values: Vec<f64>,
    v_values: Vec<f64>,
}

impl AttentionMapsTool {
    pub(super) fn serialize_state(&self) -> Option<String> {
        let seq_len = self.q_matrix.nrows();
        let d_k = self.q_matrix.ncols();
        let to_row_vec = |m: &DMatrix<f64>| {
            let mut out = Vec::with_capacity(m.nrows() * m.ncols());
            for r in 0..m.nrows() {
                for c in 0..m.ncols() {
                    out.push(m[(r, c)]);
                }
            }
            out
        };
        serde_json::to_string(&AttentionMapsState {
            seq_len,
            d_k,
            q_values: to_row_vec(&self.q_matrix),
            k_values: to_row_vec(&self.k_matrix),
            v_values: to_row_vec(&self.v_matrix),
        })
        .ok()
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        let s: AttentionMapsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid AttentionMapsTool state: {e}"))?;
        let expected = s.seq_len.checked_mul(s.d_k).unwrap_or(0);
        if s.seq_len == 0
            || s.d_k == 0
            || s.q_values.len() != expected
            || s.k_values.len() != expected
            || s.v_values.len() != expected
        {
            return Err("Invalid matrix dimensions in AttentionMapsTool state".to_string());
        }
        if s.q_values
            .iter()
            .chain(s.k_values.iter())
            .chain(s.v_values.iter())
            .any(|v| !v.is_finite())
        {
            return Err("Non-finite matrix entries in AttentionMapsTool state".to_string());
        }
        self.q_matrix = DMatrix::from_row_slice(s.seq_len, s.d_k, &s.q_values);
        self.k_matrix = DMatrix::from_row_slice(s.seq_len, s.d_k, &s.k_values);
        self.v_matrix = DMatrix::from_row_slice(s.seq_len, s.d_k, &s.v_values);
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
    "seq_len": 4,
    "d_k": 4,
    "q_values": [
      1.0, 0.0, 0.2, 0.0,
      0.0, 1.0, 0.0, 0.2,
      0.5, 0.5, 0.8, 0.0,
      0.0, 0.3, 0.0, 1.0
    ],
    "k_values": [
      1.0, 0.1, 0.0, 0.0,
      0.1, 1.0, 0.0, 0.0,
      0.4, 0.4, 1.0, 0.0,
      0.0, 0.2, 0.0, 1.0
    ],
    "v_values": [
      0.9, 0.1, 0.2, 0.0,
      0.1, 0.8, 0.1, 0.3,
      0.3, 0.4, 0.9, 0.2,
      0.0, 0.2, 0.3, 1.0
    ]
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
