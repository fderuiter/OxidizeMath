#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{default_sir_atlas_axes, is_default_sir_atlas_axes};
use super::{
    metrics, SirForkDescriptor, SirTool, MAX_SIR_DURATION, MAX_SIR_STEPS, MIN_SIR_DURATION,
    SIR_STEP_DT,
};
use crate::framework::{InteractiveTool, TraceSnapshotManager};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(serde::Serialize)]
struct SirStateRef<'a> {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    fork: Option<SirForkDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    snapshot_manager: Option<&'a TraceSnapshotManager>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    atlas_visible: bool,
    #[serde(skip_serializing_if = "is_default_sir_atlas_axes")]
    atlas_axes: [String; 2],
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SirState {
    n: f64,
    i0: f64,
    beta: f64,
    gamma: f64,
    duration: f64,
    #[serde(default)]
    fork: Option<SirForkDescriptor>,
    #[serde(default)]
    snapshot_manager: TraceSnapshotManager,
    #[serde(default)]
    atlas_visible: bool,
    #[serde(default = "default_sir_atlas_axes")]
    atlas_axes: [String; 2],
}

impl SirTool {
    pub(super) fn load_sir_state(
        &mut self,
        state: &str,
        preserve_snapshots: bool,
    ) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: SirState =
            serde_json::from_str(state).map_err(|e| format!("Invalid SIR state: {e}"))?;
        if ![s.n, s.i0, s.beta, s.gamma, s.duration]
            .iter()
            .all(|v| v.is_finite())
            || !(100.0..=100_000.0).contains(&s.n)
            || !(1.0..=s.n).contains(&s.i0)
            || !(0.0..=5.0).contains(&s.beta)
            || !(0.0..=1.0).contains(&s.gamma)
            || s.duration < 0.0
        {
            return Err("SIR parameters out of bounds".into());
        }
        if let Some(ref f) = s.fork {
            metrics::validate_sir_fork_descriptor(f, MAX_SIR_STEPS)?;
        }
        self.atlas
            .set_axis_keys(&s.atlas_axes[0], &s.atlas_axes[1])?;
        self.atlas.set_visible(s.atlas_visible);
        self.n = s.n;
        self.i0 = s.i0;
        self.beta = s.beta;
        self.gamma = s.gamma;
        self.duration = s.duration.clamp(MIN_SIR_DURATION, MAX_SIR_DURATION);
        self.branch.shadow = None;
        self.cf.is_playing = false;
        self.cf.playhead_step = None;
        self.cf.fork = s.fork;
        if let Some(f) = s.fork {
            self.cf.target_fork_day = f.fork_step as f64 * SIR_STEP_DT;
            self.cf.branch_engine.set_effective_delta(f.delta_x);
        } else {
            self.cf.branch_engine.clear();
        }
        if !preserve_snapshots {
            self.snapshot_manager = s.snapshot_manager;
        }
        self.recalculate();
        Ok(())
    }

    pub(super) fn hash_shared_parameters(&self, h: &mut impl Hasher) {
        for val in [self.n, self.i0, self.beta, self.gamma, self.duration] {
            val.to_bits().hash(h);
        }
        if let Some(ref f) = self.cf.fork {
            true.hash(h);
            f.fork_step.hash(h);
            f.baseline_n.to_bits().hash(h);
            f.baseline_i0.to_bits().hash(h);
            f.baseline_beta.to_bits().hash(h);
            f.baseline_gamma.to_bits().hash(h);
            f.delta_x.to_bits().hash(h);
        } else {
            false.hash(h);
        }
        self.atlas.is_visible().hash(h);
        self.atlas.axis_keys().hash(h);
    }

    pub(super) fn serialize_sir_state(&self, include_snapshots: bool) -> Option<String> {
        let (xk, yk) = self.atlas.axis_keys();
        let snapshot_manager = (include_snapshots && !self.snapshot_manager.is_empty())
            .then_some(&self.snapshot_manager);
        serde_json::to_string(&SirStateRef {
            n: self.n,
            i0: self.i0,
            beta: self.beta,
            gamma: self.gamma,
            duration: self.duration,
            fork: self.cf.fork,
            snapshot_manager,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        })
        .ok()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SirTool",
        display_name: "SIR Model",
        domain: "epidemiology",
        tags: &["sir", "seir", "kermack-mckendrick", "compartmental", "r0", "epidemic"],
        build: || Box::new(SirTool::default()),
    }
}

impl TheoryDescribable for SirTool {
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

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::with_capacity(5);
        map.insert(
            "N".to_string(),
            ParameterConstraint {
                min: 100.0,
                max: 100_000.0,
                step: 100.0,
            },
        );
        map.insert(
            "I0".to_string(),
            ParameterConstraint {
                min: 1.0,
                max: 50_000.0,
                step: 1.0,
            },
        );
        map.insert(
            "beta".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 5.0,
                step: 0.02,
            },
        );
        map.insert(
            "gamma".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
        );
        map.insert(
            "duration".to_string(),
            ParameterConstraint {
                min: MIN_SIR_DURATION,
                max: MAX_SIR_DURATION,
                step: 1.0,
            },
        );
        map
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

// [cite:graph_parameters_rust]
