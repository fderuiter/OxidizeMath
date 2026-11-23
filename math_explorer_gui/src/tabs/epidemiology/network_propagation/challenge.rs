//! Federated `epidemiology_hub_vaccination` and `epidemiology_bridge_quarantine` inverse
//! challenges and zero-allocation lockstep baseline twin helpers for [`NetworkPropagationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::analytics::compute_network_analytics;
use super::{ContainmentMode, NetPropState, NetworkPropagationTool, DEFAULT_CONTAINMENT_BUDGET};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use oxidize_core::rng::OxidizeRng;

/// Canonical challenge ID for budgeted super-spreader hub immunization.
pub const CHALLENGE_EPI_HUB_VACCINATION: &str = "epidemiology_hub_vaccination";
/// Canonical challenge ID for budgeted inter-cluster bridge quarantine.
pub const CHALLENGE_EPI_BRIDGE_QUARANTINE: &str = "epidemiology_bridge_quarantine";

/// Locked containment intervention budget (`5` vaccinations or edge quarantines).
pub const CHALLENGE_EPI_BUDGET: usize = DEFAULT_CONTAINMENT_BUDGET;
/// Minimum counterfactual infections averted versus the unmitigated twin required to pass.
pub const TARGET_AVERTED_INFECTIONS: usize = 10;

const HUB_NODES: usize = 72;
const HUB_BETA: f64 = 0.16;
const HUB_GAMMA: f64 = 0.05;
const HUB_SEED: u64 = 20261121;

const BRIDGE_NODES: usize = 56;
const BRIDGE_BETA: f64 = 0.22;
const BRIDGE_GAMMA: f64 = 0.04;
const BRIDGE_SEED: u64 = 20261122;

/// Counts cumulative ever-infected vertices (`Infected` + `Recovered`) with zero heap allocations.
#[must_use]
pub(crate) fn cumulative_infections(model: &NetworkEpidemicModel) -> usize {
    model
        .states()
        .iter()
        .take(model.num_nodes())
        .filter(|&&st| matches!(st, NodeState::Infected | NodeState::Recovered))
        .count()
}

impl NetworkPropagationTool {
    /// Returns `true` when an epidemiology challenge locks topology, transmission, and budget.
    #[must_use]
    pub(crate) fn is_challenge_locked(&self) -> bool {
        self.active_challenge.is_some()
    }

    /// Returns `true` when vertex vaccination is permitted under the active challenge.
    #[must_use]
    pub(crate) fn allows_vaccination(&self) -> bool {
        self.active_challenge != Some(CHALLENGE_EPI_BRIDGE_QUARANTINE)
    }

    /// Returns `true` when edge quarantine is permitted under the active challenge.
    #[must_use]
    pub(crate) fn allows_quarantine(&self) -> bool {
        self.active_challenge != Some(CHALLENGE_EPI_HUB_VACCINATION)
    }

    /// Re-initializes the unmitigated counterfactual twin on `self.seed` with preallocated buffers.
    pub(crate) fn rebuild_baseline_twin(&mut self) {
        self.baseline_rng = OxidizeRng::new(self.seed);
        let _ = self.baseline_model.set_num_nodes(self.model.num_nodes());
        let _ = self.baseline_model.set_beta(self.model.beta());
        let _ = self.baseline_model.set_gamma(self.model.gamma());
        self.baseline_model
            .initialize_geometric_graph_with_rng(&mut self.baseline_rng);
    }

    /// Advances both the mitigated network and unmitigated twin in lockstep with zero allocations.
    pub(crate) fn step_models_lockstep(&mut self) {
        self.model.step_with_rng(&mut self.rng);
        self.baseline_model.step_with_rng(&mut self.baseline_rng);
    }

    /// Exact counterfactual infections averted compared to the seeded unmitigated twin.
    #[must_use]
    pub fn counterfactual_infections_averted(&self) -> usize {
        cumulative_infections(&self.baseline_model)
            .saturating_sub(cumulative_infections(&self.model))
    }

    /// Cumulative ever-infected vertices in the seeded unmitigated twin.
    #[must_use]
    pub fn unmitigated_cumulative_infections(&self) -> usize {
        cumulative_infections(&self.baseline_model)
    }

    /// Enforces locked challenge parameters on `self` when a challenge is active.
    pub(crate) fn enforce_challenge_locks(&mut self) {
        match self.active_challenge {
            Some(CHALLENGE_EPI_HUB_VACCINATION) => {
                self.interaction_mode = ContainmentMode::VaccinateNode;
                self.apply_canonical_locks(HUB_NODES, HUB_BETA, HUB_GAMMA, HUB_SEED, true);
            }
            Some(CHALLENGE_EPI_BRIDGE_QUARANTINE) => {
                self.interaction_mode = ContainmentMode::QuarantineEdge;
                self.apply_canonical_locks(
                    BRIDGE_NODES,
                    BRIDGE_BETA,
                    BRIDGE_GAMMA,
                    BRIDGE_SEED,
                    false,
                );
            }
            _ => {}
        }
    }

    fn apply_canonical_locks(
        &mut self,
        nodes: usize,
        beta: f64,
        gamma: f64,
        seed: u64,
        vax_only: bool,
    ) {
        let topology_changed = self.model.num_nodes() != nodes
            || (self.model.beta() - beta).abs() > 1.0e-12
            || (self.model.gamma() - gamma).abs() > 1.0e-12
            || self.seed != seed;
        self.budget = CHALLENGE_EPI_BUDGET;
        if vax_only {
            self.quarantined_edges.clear();
        } else {
            self.vaccinated_nodes.clear();
        }
        self.vaccinated_nodes.truncate(CHALLENGE_EPI_BUDGET);
        self.quarantined_edges.truncate(CHALLENGE_EPI_BUDGET);
        if topology_changed {
            self.seed = seed;
            let _ = self.model.set_num_nodes(nodes);
            let _ = self.model.set_beta(beta);
            let _ = self.model.set_gamma(gamma);
            self.rebuild_seeded_graph();
        } else {
            self.enforce_budget_cap();
        }
    }

    /// Activates or clears an epidemiology containment challenge on [`NetworkPropagationTool`].
    pub(crate) fn set_epi_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(())
            }
            Some(CHALLENGE_EPI_HUB_VACCINATION) => {
                self.using_bus_topology = false;
                self.synced_bus_revision = None;
                self.active_challenge = Some(CHALLENGE_EPI_HUB_VACCINATION);
                self.enforce_challenge_locks();
                Ok(())
            }
            Some(CHALLENGE_EPI_BRIDGE_QUARANTINE) => {
                self.using_bus_topology = false;
                self.synced_bus_revision = None;
                self.active_challenge = Some(CHALLENGE_EPI_BRIDGE_QUARANTINE);
                self.enforce_challenge_locks();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for NetworkPropagationTool"
            )),
        }
    }

    /// Scores counterfactual infections averted against the seeded lockstep twin.
    #[must_use]
    pub(crate) fn eval_epi_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge?;
        let averted = self.counterfactual_infections_averted();
        let base_cum = self.unmitigated_cumulative_infections();
        let mit_cum = cumulative_infections(&self.model);
        let summary = compute_network_analytics(&self.model);
        let passed =
            self.used_budget() <= CHALLENGE_EPI_BUDGET && averted >= TARGET_AVERTED_INFECTIONS;
        let score = if passed {
            let bonus = ((averted.saturating_sub(TARGET_AVERTED_INFECTIONS)) as f64 / 20.0)
                .clamp(0.0, 1.0)
                * 10.0;
            90.0 + bonus
        } else {
            let progress = (averted as f64 / TARGET_AVERTED_INFECTIONS as f64) * 76.0;
            let prep_bonus = if self.used_budget() > 0 { 10.0 } else { 0.0 };
            (progress + prep_bonus).clamp(0.0, 88.0)
        };
        Some(ChallengeEvaluation::new(
            score,
            passed,
            "Infections averted",
            averted as f64,
            format!(
                "Averted = {averted} (need >= {TARGET_AVERTED_INFECTIONS}; twin {base_cum} vs mitigated {mit_cum}, R0 = {:.2})",
                summary.effective_r0
            ),
        ))
    }
}

fn generate_epi_baseline(nodes: usize, beta: f64, gamma: f64, seed: u64) -> String {
    serde_json::to_string(&NetPropState {
        num_nodes: nodes,
        beta,
        gamma,
        seed,
        budget: CHALLENGE_EPI_BUDGET,
        vaccinated_nodes: Vec::new(),
        quarantined_edges: Vec::new(),
        is_running: false,
    })
    .unwrap_or_default()
}

fn generate_hub_baseline() -> String {
    generate_epi_baseline(HUB_NODES, HUB_BETA, HUB_GAMMA, HUB_SEED)
}

fn generate_bridge_baseline() -> String {
    generate_epi_baseline(BRIDGE_NODES, BRIDGE_BETA, BRIDGE_GAMMA, BRIDGE_SEED)
}

fn sanitize_usize_array(val: Option<&serde_json::Value>) -> Vec<usize> {
    let mut out = Vec::new();
    if let Some(arr) = val.and_then(serde_json::Value::as_array) {
        for item in arr {
            if let Some(idx) = item.as_u64().map(|n| n as usize) {
                if !out.contains(&idx) {
                    out.push(idx);
                }
            }
        }
    }
    out
}

fn clamp_epi_challenge_state(
    raw_json: &str,
    challenge_id: &'static str,
    nodes: usize,
    beta: f64,
    gamma: f64,
    seed: u64,
    vax_only: bool,
) -> Result<String, String> {
    let val: serde_json::Value = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid NetworkPropagationTool challenge JSON: {e}"))?;
    let obj = val.as_object().ok_or_else(|| {
        "NetworkPropagationTool challenge state must be a JSON object".to_string()
    })?;

    let mut probe_rng = OxidizeRng::new(seed);
    let probe = NetworkEpidemicModel::new_with_rng(nodes, beta, gamma, &mut probe_rng);
    let vaccinated_nodes = if vax_only {
        sanitize_usize_array(obj.get("vaccinated_nodes"))
            .into_iter()
            .filter(|&n| {
                n < nodes
                    && probe.patient_zero() != Some(n)
                    && probe.states()[n] == NodeState::Susceptible
            })
            .take(CHALLENGE_EPI_BUDGET)
            .collect()
    } else {
        Vec::new()
    };
    let quarantined_edges = if vax_only {
        Vec::new()
    } else {
        let edge_len = probe.edges().len();
        sanitize_usize_array(obj.get("quarantined_edges"))
            .into_iter()
            .filter(|&e| e < edge_len)
            .take(CHALLENGE_EPI_BUDGET)
            .collect()
    };
    let is_running = obj
        .get("is_running")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let clamped = serde_json::to_string(&NetPropState {
        num_nodes: nodes,
        beta,
        gamma,
        seed,
        budget: CHALLENGE_EPI_BUDGET,
        vaccinated_nodes,
        quarantined_edges,
        is_running,
    })
    .map_err(|e| e.to_string())?;

    let mut tool = NetworkPropagationTool::default();
    tool.set_epi_challenge(Some(challenge_id))?;
    tool.try_load_state(&clamped)?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped NetworkPropagationTool state".to_string())
}

fn clamp_hub_state(raw_json: &str) -> Result<String, String> {
    clamp_epi_challenge_state(
        raw_json,
        CHALLENGE_EPI_HUB_VACCINATION,
        HUB_NODES,
        HUB_BETA,
        HUB_GAMMA,
        HUB_SEED,
        true,
    )
}

fn clamp_bridge_state(raw_json: &str) -> Result<String, String> {
    clamp_epi_challenge_state(
        raw_json,
        CHALLENGE_EPI_BRIDGE_QUARANTINE,
        BRIDGE_NODES,
        BRIDGE_BETA,
        BRIDGE_GAMMA,
        BRIDGE_SEED,
        false,
    )
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_EPI_HUB_VACCINATION,
        title: "Super-Spreader Hub Vaccination",
        description: "Immunize up to 5 susceptible hubs on a seeded 72-node contact graph and step the outbreak to avert >= 10 infections versus the unmitigated twin.",
        category: "Network Epidemiology",
        icon: "💉",
        domain: "epidemiology",
        tool_name: "NetworkPropagationTool",
        locked_inputs: "N = 72, β = 0.16, γ = 0.05, seed = 20261121, budget = 5 (vaccinations only)",
        tunable_budget: "Up to 5 vertex vaccinations on susceptible non-patient-zero nodes",
        target_summary: "Avert >= 10 cumulative infections against the lockstep unmitigated twin",
        baseline_state_generator: generate_hub_baseline,
        clamp_locked_state: clamp_hub_state,
    }
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_EPI_BRIDGE_QUARANTINE,
        title: "Community Bridge Quarantine",
        description: "Quarantine up to 5 inter-cluster contact edges on a seeded 56-node graph and step the outbreak to avert >= 10 infections versus the unmitigated twin.",
        category: "Network Epidemiology",
        icon: "✂️",
        domain: "epidemiology",
        tool_name: "NetworkPropagationTool",
        locked_inputs: "N = 56, β = 0.22, γ = 0.04, seed = 20261122, budget = 5 (edge quarantines only)",
        tunable_budget: "Up to 5 undirected contact-edge quarantines",
        target_summary: "Avert >= 10 cumulative infections against the lockstep unmitigated twin",
        baseline_state_generator: generate_bridge_baseline,
        clamp_locked_state: clamp_bridge_state,
    }
}

// [cite:algorithmic_information_rust]
// [cite:graph_parameters_rust]
