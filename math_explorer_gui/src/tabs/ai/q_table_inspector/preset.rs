const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Q-Table Inspector",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "epsilon": 0.1,
    "alpha": 0.2
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "q_table_inspector_starter",
        title: "Q-Table Inspector",
        description: "Tabular temporal difference state-action value matrix visualization and convergence tracking.",
        category: "Reinforcement Learning",
        icon: "📊",
        domain: "ai",
        tool_name: Some("QTableInspectorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
