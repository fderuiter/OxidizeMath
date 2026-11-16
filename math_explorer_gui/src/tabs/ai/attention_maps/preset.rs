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
