//! Metadata, theory description, and starter presets for [`super::SirTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SirTool;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SirTool",
        display_name: "SIR Model",
        domain: "epidemiology",
        tags: &[],
        build: || Box::new(SirTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SirTool {
    fn theory_description(&self) -> String {
        let r0 = self.beta / self.gamma;
        format!(
            "Kermack-McKendrick SIR compartmental epidemic model: $\\frac{{dS}}{{dt}} = -\\frac{{\\beta S I}}{{N}}$, $\\frac{{dI}}{{dt}} = \\frac{{\\beta S I}}{{N}} - \\gamma I$, $\\frac{{dR}}{{dt}} = \\gamma I$. Transmission rate $\\beta = {:.3}$, recovery rate $\\gamma = {:.3}$, basic reproduction number $R_0 = {:.2}$.",
            self.beta, self.gamma, r0
        )
    }

    fn phonetic_description(&self) -> String {
        "Kermack-McKendrick SIR compartmental epidemic model theory".into()
    }

    fn theory_citation(&self) -> String {
        "A Contribution to the Mathematical Theory of Epidemics (Kermack & McKendrick, 1927)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "Kermack-McKendrick SIR Model", "domain": "epidemiology", "timestamp": "2026-11-16T00:00:00Z" },
  "config": { "n": 10000.0, "i0": 25.0, "beta": 0.3, "gamma": 0.1, "duration": 120.0 }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sir_starter",
        title: "Kermack-McKendrick SIR Model",
        description: "dS/dt = -βSI/N, dI/dt = βSI/N - γI: Compartmental epidemic outbreak dynamics and herd immunity.",
        category: "Compartmental Models",
        icon: "🦠",
        domain: "epidemiology",
        tool_name: Some("SirTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sir_lockdown_timing",
        title: "SIR Lockdown Timing",
        description: "Counterfactual shadow twin forked at day 10 lowering β from 0.5 to 0.2 to measure ΔI_peak infections averted.",
        category: "Counterfactual Epidemiology",
        icon: "⑂",
        domain: "epidemiology",
        tool_name: Some("SirTool"),
        preset_generator: || crate::starter_gallery::PRESET_SIR_LOCKDOWN_TIMING,
    }
}
