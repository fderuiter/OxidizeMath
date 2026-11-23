#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::LoraHubSimplexTool;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LoraHubSimplexTool",
        display_name: "LoraHub Simplex Composer",
        domain: "ai",
        tags: &["ai", "lora", "lorahub", "optimization", "simplex"],
        build: || Box::new(LoraHubSimplexTool::default()),
    }
}

const LORAHUB_SPARSE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "LoraHub Sparse Adapter Pruning",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "alpha": 0.55,
    "initial_step": 0.42,
    "weight_bound": 1.5,
    "task_preset": "math_code_synthesis"
  }
}"#;

const LORAHUB_CROSS_LINGUAL_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "LoraHub Cross-Lingual Blend",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "alpha": 0.2,
    "initial_step": 0.38,
    "weight_bound": 1.5,
    "task_preset": "cross_lingual_summary"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lorahub_sparse_adapter_pruning",
        title: "LoraHub Sparse Adapter Pruning",
        description: "Gradient-free Nelder-Mead LoRA module composition with L1 sparsity suppressing non-essential adapters.",
        category: "Parameter-Efficient Adaptation",
        icon: "🧩",
        domain: "ai",
        tool_name: Some("LoraHubSimplexTool"),
        preset_generator: || LORAHUB_SPARSE_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lorahub_cross_lingual_blend",
        title: "LoraHub Cross-Lingual Blend",
        description: "Signed linear combination of pre-trained LoRA state dictionaries discovered via black-box simplex reflections.",
        category: "Parameter-Efficient Adaptation",
        icon: "🔀",
        domain: "ai",
        tool_name: Some("LoraHubSimplexTool"),
        preset_generator: || LORAHUB_CROSS_LINGUAL_PRESET_JSON,
    }
}
