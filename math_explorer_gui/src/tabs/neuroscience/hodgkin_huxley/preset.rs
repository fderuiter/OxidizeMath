#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::HodgkinHuxleyUnified;
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use math_explorer::biology::neuroscience::{HodgkinHuxleyModel, HodgkinHuxleyParameters};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "HodgkinHuxleyTool",
        display_name: "Hodgkin-Huxley Model",
        domain: "neuroscience",
        tags: &["action potential", "conductance", "ion channel", "membrane", "neuron"],
        build: || Box::new(UnifiedSimTool::<HodgkinHuxleyUnified>::new()),
    }
}

impl TheoryDescribable for HodgkinHuxleyUnified {
    fn theory_description(&self) -> String {
        HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 0.0).theory_description()
    }

    fn phonetic_description(&self) -> String {
        HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 0.0).phonetic_description()
    }

    fn theory_citation(&self) -> String {
        HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 0.0).theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HodgkinHuxleyModel::new(HodgkinHuxleyParameters::default(), 0.0).available_descriptions()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        Self::parameters()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hodgkin-Huxley Action Potential",
    "domain": "neuroscience",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "g_na": 120.0,
    "g_k": 36.0,
    "g_l": 0.3,
    "i_ext": 10.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "hodgkin_huxley_starter",
        title: "Hodgkin-Huxley Action Potential",
        description: "C_m·dV/dt = -g_Na·m³h(V - E_Na) - g_K·n⁴(V - E_K) - g_L(V - E_L) + I: Squid axon membrane spikes.",
        category: "Computational Neuroscience",
        icon: "🧠",
        domain: "neuroscience",
        tool_name: Some("HodgkinHuxleyTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
