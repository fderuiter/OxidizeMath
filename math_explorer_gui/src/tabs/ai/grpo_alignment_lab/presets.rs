#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GrpoAlignmentLabTool;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "GrpoAlignmentLabTool",
        display_name: "GRPO Alignment Lab",
        domain: "ai",
        tags: &["ai", "rl", "grpo", "alignment", "calibration"],
        build: || Box::new(GrpoAlignmentLabTool::default()),
    }
}

const GRPO_SURROGATE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "GRPO Frontier Curriculum",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "epsilon": 0.2,
    "beta": 0.04,
    "solver_accuracy": 0.5,
    "repetition_lambda": 0.45,
    "cluster_size": 2,
    "confidence_skew": 0.65,
    "scenario": "reasoning_consensus"
  }
}"#;

const GRPO_CALIBRATION_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Self-Calibration Entropy Distillation",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "epsilon": 0.15,
    "beta": 0.08,
    "solver_accuracy": 0.42,
    "repetition_lambda": 0.8,
    "cluster_size": 3,
    "confidence_skew": 0.8,
    "scenario": "high_entropy_exploration"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "grpo_frontier_curriculum",
        title: "GRPO Frontier Curriculum",
        description: "Group Relative Policy Optimization clipped surrogate and Challenger peak-uncertainty curriculum at 50% solver accuracy.",
        category: "Alignment & Calibration",
        icon: "⚖",
        domain: "ai",
        tool_name: Some("GrpoAlignmentLabTool"),
        preset_generator: || GRPO_SURROGATE_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "grpo_self_calibration_distillation",
        title: "Self-Calibration Entropy Distillation",
        description: "Dynamic answer-entropy temperature scaling and KL confidence distillation across multi-path reasoning candidates.",
        category: "Alignment & Calibration",
        icon: "🎯",
        domain: "ai",
        tool_name: Some("GrpoAlignmentLabTool"),
        preset_generator: || GRPO_CALIBRATION_PRESET_JSON,
    }
}
