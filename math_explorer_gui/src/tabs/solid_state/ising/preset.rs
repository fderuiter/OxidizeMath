#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::IsingModelTool;

/// Lattice evolution kernel selection for [`IsingModelTool`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum IsingInteractionMode {
    /// Classical scalar ferromagnetic/antiferromagnetic Metropolis-Hastings dynamics.
    #[default]
    Ferromagnetic,
    /// Spatial 2x2 evolutionary game with pairwise Fermi imitation.
    FermiGame,
}

/// Canonical 2x2 evolutionary game presets `(R, S, T, P)` for spatial Fermi dynamics.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum FermiGamePreset {
    /// Prisoner's Dilemma (`T > R > P > S`): boundary reciprocity sustains cooperator droplets while 0D collapses.
    #[default]
    PrisonersDilemma,
    /// Stag-Hunt (`R > T > P > S`): bistable coordination game with critical nucleation threshold.
    StagHunt,
    /// Hawk-Dove / Snowdrift (`T > R > S > P`): polymorphic coexistence of cooperators and defectors.
    HawkDove,
    /// User-edited 2x2 payoff matrix.
    Custom,
}

impl FermiGamePreset {
    /// Display label for preset buttons and combo selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PrisonersDilemma => "Prisoner's Dilemma",
            Self::StagHunt => "Stag-Hunt",
            Self::HawkDove => "Hawk-Dove",
            Self::Custom => "Custom",
        }
    }

    /// Canonical `[R, S, T, P]` 2x2 payoff matrix entries (`+1` Cooperator = row 0, `-1` Defector = row 1).
    ///
    /// `Self::HawkDove` derives its `[R, S, T, P]` entries directly from
    /// [`math_explorer::biology::evolution::HawkDovePopulation`] (`V = 2.0, C = 4.0`, where
    /// Cooperator = Dove and Defector = Hawk).
    #[must_use]
    pub fn payoffs(self) -> [f64; 4] {
        match self {
            Self::PrisonersDilemma | Self::Custom => [1.0, -0.1, 1.25, 0.0],
            Self::StagHunt => [2.0, 0.0, 1.2, 1.0],
            Self::HawkDove => math_explorer::biology::evolution::HawkDovePopulation::new(2.0, 4.0)
                .cooperator_defector_payoffs(),
        }
    }

    /// Recommended rationality noise temperature $T$ for the preset.
    #[must_use]
    pub const fn default_temperature(self) -> f64 {
        match self {
            Self::PrisonersDilemma => 0.08,
            Self::StagHunt | Self::HawkDove | Self::Custom => 0.15,
        }
    }
}

pub(super) fn default_game_payoffs() -> [f64; 4] {
    FermiGamePreset::PrisonersDilemma.payoffs()
}

pub(super) const fn default_brush_radius() -> i32 {
    6
}

pub(super) const fn default_brush_cooperator() -> bool {
    true
}

fn is_zero_f64(val: &f64) -> bool {
    val.abs() <= f64::EPSILON
}

const fn default_true() -> bool {
    true
}

fn is_true(val: &bool) -> bool {
    *val
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct IsingState {
    pub(crate) temperature: f64,
    pub(crate) j_coupling: f64,
    pub(crate) h_field: f64,
    #[serde(default)]
    pub(crate) mode: IsingInteractionMode,
    #[serde(default)]
    pub(crate) game_preset: FermiGamePreset,
    #[serde(default = "default_game_payoffs")]
    pub(crate) game_payoffs: [f64; 4],
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub(crate) mutation_rate: f64,
    #[serde(default = "default_brush_radius")]
    pub(crate) brush_radius: i32,
    #[serde(default = "default_brush_cooperator")]
    pub(crate) brush_cooperator: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(crate) defector_sea: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) droplet_seeds: Vec<super::DropletSeed>,
    #[serde(
        default,
        skip_serializing_if = "super::atlas::IsingAtlasPersisted::is_default"
    )]
    pub(crate) atlas: super::atlas::IsingAtlasPersisted,
    #[serde(default)]
    pub(crate) state: Option<crate::async_sim::SimWorkerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) running: Option<bool>,
    pub(crate) steps_per_frame: usize,
    #[serde(
        default,
        skip_serializing_if = "crate::app::TopologySource::is_default"
    )]
    pub(crate) topology_source: crate::app::TopologySource,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub(crate) follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) seeded_graph: Option<crate::app::data_bus::SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) frozen_graph: Option<crate::app::data_bus::SharedGraph>,
}

impl IsingState {
    pub(crate) fn validate(&self) -> Result<(), String> {
        let seeds_valid = self.droplet_seeds.len() <= super::MAX_DROPLET_SEEDS
            && self.droplet_seeds.iter().all(|s| {
                (0..=99).contains(&s.cx)
                    && (0..=99).contains(&s.cy)
                    && (0..=99).contains(&s.r)
                    && (s.spin == 1 || s.spin == -1)
            });
        let valid = self.temperature.is_finite()
            && (1e-3..=10.0).contains(&self.temperature)
            && self.j_coupling.is_finite()
            && (-2.0..=2.0).contains(&self.j_coupling)
            && self.h_field.is_finite()
            && (-2.0..=2.0).contains(&self.h_field)
            && self.mutation_rate.is_finite()
            && (0.0..=0.05).contains(&self.mutation_rate)
            && self
                .game_payoffs
                .iter()
                .all(|v| v.is_finite() && (-20.0..=20.0).contains(v))
            && (0..=99).contains(&self.brush_radius)
            && (100..=50_000).contains(&self.steps_per_frame)
            && seeds_valid;
        if !valid {
            return Err("Ising parameters out of valid bounds".to_string());
        }
        for g in [&self.seeded_graph, &self.frozen_graph]
            .into_iter()
            .flatten()
        {
            g.validate(
                crate::app::data_bus::SESSION_MAX_BUS_NODES,
                crate::app::data_bus::SESSION_MAX_BUS_EDGES,
            )?;
        }
        Ok(())
    }
}

impl IsingModelTool {
    pub(super) fn serialize_ising_state(&self) -> Option<String> {
        let (xk, yk) = self.atlas.axis_keys();
        let frozen_graph = (!self.graph_mode.follow_workspace_graph)
            .then(|| self.graph_mode.active_graph.as_ref().map(|g| (**g).clone()))
            .flatten();
        serde_json::to_string(&IsingState {
            temperature: self.temperature,
            j_coupling: self.j_coupling,
            h_field: self.h_field,
            mode: self.mode,
            game_preset: self.game_preset,
            game_payoffs: self.game_payoffs,
            mutation_rate: self.mutation_rate,
            brush_radius: self.brush_radius,
            brush_cooperator: self.brush_cooperator,
            defector_sea: self.defector_sea,
            droplet_seeds: self.droplet_seeds.clone(),
            atlas: super::atlas::IsingAtlasPersisted {
                visible: self.atlas.is_visible(),
                axes: [xk.to_string(), yk.to_string()],
            },
            state: Some(self.worker.state()),
            running: None,
            steps_per_frame: self.steps_per_frame,
            topology_source: self.topology_source,
            follow_workspace_graph: self.graph_mode.follow_workspace_graph,
            seeded_graph: self.graph_mode.seeded_graph.clone(),
            frozen_graph,
        })
        .ok()
    }

    pub(super) fn deserialize_ising_state(&mut self, state: &str) -> Result<(), String> {
        use crate::framework::InteractiveTool;
        use std::sync::Arc;
        self.state_bounds().validate_json(state)?;
        let s: IsingState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Ising state: {e}"))?;
        s.validate()?;
        self.atlas
            .set_axis_keys(&s.atlas.axes[0], &s.atlas.axes[1])?;
        self.atlas.set_visible(s.atlas.visible);
        self.temperature = s.temperature;
        self.j_coupling = s.j_coupling;
        self.h_field = s.h_field;
        self.mutation_rate = s.mutation_rate;
        self.mode = s.mode;
        self.game_preset = s.game_preset;
        self.game_payoffs = s.game_payoffs;
        self.brush_radius = s.brush_radius;
        self.brush_cooperator = s.brush_cooperator;
        self.defector_sea = s.defector_sea;
        self.droplet_seeds = s.droplet_seeds;
        self.steps_per_frame = s.steps_per_frame;
        self.topology_source = s.topology_source;
        self.graph_mode.follow_workspace_graph = s.follow_workspace_graph;
        self.graph_mode.seeded_graph = s.seeded_graph.clone();
        self.graph_mode.pending_seed_publish =
            s.seeded_graph.is_some() && s.topology_source == crate::app::TopologySource::SharedBus;
        if let Some(g) = s.frozen_graph.or(s.seeded_graph) {
            self.apply_shared_spin_graph(&Arc::new(g));
        } else if self.graph_mode.follow_workspace_graph
            || self.topology_source == crate::app::TopologySource::LocalPreset
        {
            self.detach_shared_csr_graph(false);
        }
        self.enforce_ising_challenge_locks();
        self.sync_shared_params();
        if self.defector_sea || !self.droplet_seeds.is_empty() {
            self.push_seed_layout_to_worker();
        }
        self.recompute_replicator_curve();
        self.refresh_challenge_workspace();
        self.sync_worker_target_state(s.state, s.running);
        self.stepped_pixels = None;
        self.dirty = true;
        Ok(())
    }

    pub(super) fn record_droplet_seed(&mut self, seed: super::DropletSeed) {
        if self.droplet_seeds.last() == Some(&seed) {
            return;
        }
        let cap = if self.active_challenge.is_some() {
            4
        } else {
            super::MAX_DROPLET_SEEDS
        };
        if self.droplet_seeds.len() >= cap {
            self.droplet_seeds.remove(0);
        }
        self.droplet_seeds.push(seed);
    }
}

impl scientific_metadata::theory::TheoryDescribable for IsingModelTool {
    fn theory_description(&self) -> String {
        "Two-dimensional square lattice ferromagnetic Ising model governed by Hamiltonian $H = -J \\sum_{\\langle i, j \\rangle} s_i s_j - h \\sum_i s_i$ with Onsager critical temperature $T_c = \\frac{2J}{k_B \\ln(1 + \\sqrt{2})} \\approx 2.269 \\frac{J}{k_B}$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Two-dimensional Ising model ferromagnetic phase transition theory".into()
    }
    fn theory_citation(&self) -> String {
        "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition (Onsager, 1944)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
    fn theory_parameters(
        &self,
    ) -> std::collections::HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        use scientific_metadata::theory::ParameterConstraint;
        [
            ("temperature", 0.1, 6.0, 0.05),
            ("j_coupling", -2.0, 2.0, 0.02),
            ("h_field", -2.0, 2.0, 0.02),
            ("mutation_rate", 0.0, 0.05, 0.001),
            ("T", 0.0, 2.0, 0.02),
            ("S", -1.0, 1.0, 0.02),
        ]
        .into_iter()
        .map(|(k, min, max, step)| (k.to_string(), ParameterConstraint { min, max, step }))
        .collect()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "2D Ising Model Ferromagnetism",
    "domain": "solid_state",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "temperature": 2.269,
    "j_coupling": 1.0,
    "h_field": 0.05,
    "steps_per_frame": 10000
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "ising_model_starter",
        title: "2D Ising Model Ferromagnetism",
        description: "H = -J ∑ s_i·s_j - h ∑ s_i: Monte Carlo Metropolis spin flip dynamics and Curie transition.",
        category: "Statistical Mechanics",
        icon: "🧲",
        domain: "solid_state",
        tool_name: Some("IsingModelTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

const FRUSTRATED_TRIANGULAR_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Frustrated Triangular Spins",
    "domain": "solid_state",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "temperature": 0.45,
    "j_coupling": 1.0,
    "h_field": 0.0,
    "steps_per_frame": 2000,
    "seeded_graph": {
      "revision": 1,
      "nodes": [
        { "id": 0, "norm_pos": [0.50, 0.50] },
        { "id": 1, "norm_pos": [0.50, 0.14] },
        { "id": 2, "norm_pos": [0.84, 0.36] },
        { "id": 3, "norm_pos": [0.72, 0.78] },
        { "id": 4, "norm_pos": [0.28, 0.78] },
        { "id": 5, "norm_pos": [0.16, 0.36] }
      ],
      "edges": [
        [0, 1, -1.0],
        [0, 2, -1.0],
        [0, 3, -1.0],
        [0, 4, -1.0],
        [0, 5, -1.0],
        [1, 2, -1.0],
        [2, 3, -1.0],
        [3, 4, -1.0],
        [4, 5, -1.0],
        [5, 1, -1.0]
      ]
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "frustrated_triangular_spins",
        title: "Frustrated Triangular Spins",
        description: "Odd-cycle antiferromagnetic (-w) wheel graph exhibiting geometric spin frustration with no zero-frustration ground state.",
        category: "Statistical Mechanics",
        icon: "🔺",
        domain: "solid_state",
        tool_name: Some("IsingModelTool"),
        preset_generator: || FRUSTRATED_TRIANGULAR_PRESET_JSON,
    }
}
