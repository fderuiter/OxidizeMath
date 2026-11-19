#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::AttentionMapsTool;
use nalgebra::DMatrix;

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
