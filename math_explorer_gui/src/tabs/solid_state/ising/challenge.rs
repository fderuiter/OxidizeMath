//! Preallocated zero-heap-allocation nucleation workspace and `solid_state` inverse challenges
//! (`ising_critical_droplet_nucleation` and `fermi_prisoners_dilemma_nucleation`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::{FermiGamePreset, IsingInteractionMode};
use super::runner::{DropletSeed, MAX_DROPLET_SEEDS, apply_circular_droplet};
use super::{DEFAULT_ISING_SIZE, IsingModelTool, IsingState};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};
use math_explorer::applied::game_theory::evolutionary::MatrixPayoff;
use math_explorer::physics::stat_mech::KB;
use math_explorer::physics::stat_mech::ising::SpinLattice;
use nalgebra::DMatrix;

/// Classical Becker-Döring critical droplet nucleation challenge ID.
pub const CHALLENGE_ISING_DROPLET_NUCLEATION: &str = "ising_critical_droplet_nucleation";

/// Spatial Prisoner's Dilemma cooperator cluster nucleation challenge ID.
pub const CHALLENGE_FERMI_PD_NUCLEATION: &str = "fermi_prisoners_dilemma_nucleation";

/// Maximum initial seeded `+1` spin sites permitted in nucleation challenges (`3.8%` of `100x100`).
pub const NUCLEATION_MAX_SEED_SITES: usize = 380;
/// Deterministic Monte Carlo trial count (`45` sweeps = `450,000` trials) for challenge verification.
const NUCLEATION_VERIFY_TRIALS: usize = 450_000;
const ISING_NUCLEATION_SEED: u64 = 0x1516_C217;
const FERMI_NUCLEATION_SEED: u64 = 0xFE24_1C00;

/// Summary metrics produced by [`IsingChallengeWorkspace`] with zero dynamic heap allocations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IsingNucleationSummary {
    /// Initial `+1` spin count painted onto the `-1` sea before evolution.
    pub initial_plus_sites: usize,
    /// Final `+1` fraction in `[0.0, 1.0]` after deterministic Metropolis / Fermi evolution.
    pub final_plus_fraction: f64,
    /// Normalized challenge score in `0.0..=100.0`.
    pub score: f64,
    /// Whether both the seed-budget cap and supercritical growth target hold.
    pub passed: bool,
}

/// Preallocated `100x100` [`SpinLattice`] workspace verifying droplet nucleation with `0` allocations.
pub struct IsingChallengeWorkspace {
    lattice: SpinLattice,
    pd_payoff: MatrixPayoff,
    pub summary: IsingNucleationSummary,
}

impl Default for IsingChallengeWorkspace {
    fn default() -> Self {
        let mat = DMatrix::from_row_slice(2, 2, &FermiGamePreset::PrisonersDilemma.payoffs());
        let pd_payoff = MatrixPayoff::new(mat).expect("valid 2x2 Prisoner's Dilemma payoff");
        Self {
            lattice: SpinLattice::new(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, Some(42)),
            pd_payoff,
            summary: IsingNucleationSummary::default(),
        }
    }
}

impl IsingChallengeWorkspace {
    /// Evaluates classical Ising droplet nucleation (`T = 1.2, J = 1.0, h = 0.35`) with `0` allocations.
    pub fn evaluate_ising_nucleation(&mut self, seeds: &[DropletSeed]) -> IsingNucleationSummary {
        self.lattice
            .reset_uniform_with_seed(-1, ISING_NUCLEATION_SEED);
        for s in seeds.iter().take(MAX_DROPLET_SEEDS) {
            let r = s.r.clamp(0, 12);
            apply_circular_droplet(
                &mut self.lattice,
                s.cx,
                s.cy,
                r,
                if s.spin >= 0 { 1 } else { -1 },
            );
        }
        let total = (self.lattice.width() * self.lattice.height()).max(1);
        let init_m = self.lattice.magnetization();
        let initial_plus_sites = ((init_m + total as i64) / 2).max(0) as usize;

        let t_phys = 1.2 / KB;
        self.lattice
            .evolve(NUCLEATION_VERIFY_TRIALS, t_phys, 1.0, 0.35);
        let final_m = self.lattice.magnetization();
        let final_plus_fraction = (((final_m as f64 / total as f64) + 1.0) * 0.5).clamp(0.0, 1.0);

        let target_frac = 0.06;
        let within_budget = (1..=NUCLEATION_MAX_SEED_SITES).contains(&initial_plus_sites);
        let passed = within_budget && final_plus_fraction >= target_frac;
        let growth_score = (100.0 * (final_plus_fraction / target_frac)).clamp(0.0, 100.0);
        let score = if within_budget {
            growth_score
        } else if initial_plus_sites == 0 {
            0.0
        } else {
            let over = (initial_plus_sites - NUCLEATION_MAX_SEED_SITES) as f64;
            (growth_score - 0.5 * over).clamp(0.0, 45.0)
        };
        let summary = IsingNucleationSummary {
            initial_plus_sites,
            final_plus_fraction,
            score,
            passed,
        };
        self.summary = summary;
        summary
    }

    /// Evaluates Prisoner's Dilemma spatial cooperator cluster nucleation with `0` allocations.
    pub fn evaluate_fermi_nucleation(&mut self, seeds: &[DropletSeed]) -> IsingNucleationSummary {
        self.lattice
            .reset_uniform_with_seed(-1, FERMI_NUCLEATION_SEED);
        for s in seeds.iter().take(MAX_DROPLET_SEEDS) {
            let r = s.r.clamp(0, 12);
            apply_circular_droplet(
                &mut self.lattice,
                s.cx,
                s.cy,
                r,
                if s.spin >= 0 { 1 } else { -1 },
            );
        }
        let total = (self.lattice.width() * self.lattice.height()).max(1);
        let init_m = self.lattice.magnetization();
        let initial_plus_sites = ((init_m + total as i64) / 2).max(0) as usize;

        self.lattice.evolve_fermi_game_with_mutation(
            NUCLEATION_VERIFY_TRIALS,
            0.08,
            &self.pd_payoff,
            0.0,
        );
        let final_m = self.lattice.magnetization();
        let final_plus_fraction = (((final_m as f64 / total as f64) + 1.0) * 0.5).clamp(0.0, 1.0);

        let target_frac = 0.08;
        let within_budget = (1..=NUCLEATION_MAX_SEED_SITES).contains(&initial_plus_sites);
        let passed = within_budget && final_plus_fraction >= target_frac;
        let growth_score = (100.0 * (final_plus_fraction / target_frac)).clamp(0.0, 100.0);
        let score = if within_budget {
            growth_score
        } else if initial_plus_sites == 0 {
            0.0
        } else {
            let over = (initial_plus_sites - NUCLEATION_MAX_SEED_SITES) as f64;
            (growth_score - 0.5 * over).clamp(0.0, 45.0)
        };
        let summary = IsingNucleationSummary {
            initial_plus_sites,
            final_plus_fraction,
            score,
            passed,
        };
        self.summary = summary;
        summary
    }
}

impl IsingModelTool {
    pub(crate) fn enforce_ising_challenge_locks(&mut self) {
        if self.active_challenge.is_some() {
            self.detach_shared_csr_graph(false);
        }
        match self.active_challenge {
            Some(CHALLENGE_ISING_DROPLET_NUCLEATION) => {
                self.mode = IsingInteractionMode::Ferromagnetic;
                self.temperature = 1.2;
                self.j_coupling = 1.0;
                self.h_field = 0.35;
                self.mutation_rate = 0.0;
                self.defector_sea = true;
                self.brush_radius = self.brush_radius.clamp(0, 10);
                for s in &mut self.droplet_seeds {
                    s.cx = s.cx.clamp(0, 99);
                    s.cy = s.cy.clamp(0, 99);
                    s.r = s.r.clamp(0, 10);
                    s.spin = if s.spin >= 0 { 1 } else { -1 };
                }
                self.droplet_seeds.truncate(4);
            }
            Some(CHALLENGE_FERMI_PD_NUCLEATION) => {
                self.mode = IsingInteractionMode::FermiGame;
                self.game_preset = FermiGamePreset::PrisonersDilemma;
                self.game_payoffs = FermiGamePreset::PrisonersDilemma.payoffs();
                self.temperature = 0.08;
                self.mutation_rate = 0.0;
                self.defector_sea = true;
                self.brush_radius = self.brush_radius.clamp(0, 10);
                for s in &mut self.droplet_seeds {
                    s.cx = s.cx.clamp(0, 99);
                    s.cy = s.cy.clamp(0, 99);
                    s.r = s.r.clamp(0, 10);
                    s.spin = if s.spin >= 0 { 1 } else { -1 };
                }
                self.droplet_seeds.truncate(4);
            }
            _ => {}
        }
    }

    pub(crate) fn refresh_challenge_workspace(&mut self) {
        match self.active_challenge {
            Some(CHALLENGE_ISING_DROPLET_NUCLEATION) => {
                self.challenge_workspace
                    .evaluate_ising_nucleation(&self.droplet_seeds);
            }
            Some(CHALLENGE_FERMI_PD_NUCLEATION) => {
                self.challenge_workspace
                    .evaluate_fermi_nucleation(&self.droplet_seeds);
            }
            _ => {}
        }
    }

    pub(crate) fn set_ising_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(())
            }
            Some(CHALLENGE_ISING_DROPLET_NUCLEATION) => {
                self.active_challenge = Some(CHALLENGE_ISING_DROPLET_NUCLEATION);
                self.enforce_ising_challenge_locks();
                self.sync_shared_params();
                self.push_seed_layout_to_worker();
                self.refresh_challenge_workspace();
                Ok(())
            }
            Some(CHALLENGE_FERMI_PD_NUCLEATION) => {
                self.active_challenge = Some(CHALLENGE_FERMI_PD_NUCLEATION);
                self.enforce_ising_challenge_locks();
                self.sync_shared_params();
                self.push_seed_layout_to_worker();
                self.recompute_replicator_curve();
                self.refresh_challenge_workspace();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for IsingModelTool"
            )),
        }
    }

    pub(crate) fn eval_ising_challenge(&self) -> Option<ChallengeEvaluation> {
        let s = &self.challenge_workspace.summary;
        match self.active_challenge {
            Some(CHALLENGE_ISING_DROPLET_NUCLEATION) => {
                let shortfall = (0.06 - s.final_plus_fraction).max(0.0);
                let budget_over = s
                    .initial_plus_sites
                    .saturating_sub(NUCLEATION_MAX_SEED_SITES);
                Some(ChallengeEvaluation::new(
                    s.score,
                    s.passed,
                    "Growth Deficit",
                    shortfall + budget_over as f64,
                    format!(
                        "Seeded = {}/{} sites, Final +1 = {:.1}% (need ≥ 6.0%)",
                        s.initial_plus_sites,
                        NUCLEATION_MAX_SEED_SITES,
                        s.final_plus_fraction * 100.0
                    ),
                ))
            }
            Some(CHALLENGE_FERMI_PD_NUCLEATION) => {
                let shortfall = (0.08 - s.final_plus_fraction).max(0.0);
                let budget_over = s
                    .initial_plus_sites
                    .saturating_sub(NUCLEATION_MAX_SEED_SITES);
                Some(ChallengeEvaluation::new(
                    s.score,
                    s.passed,
                    "Cooperator Deficit",
                    shortfall + budget_over as f64,
                    format!(
                        "Seeded = {}/{} sites, Spatial C = {:.1}% (need ≥ 8.0%, 0D ODE → 0%)",
                        s.initial_plus_sites,
                        NUCLEATION_MAX_SEED_SITES,
                        s.final_plus_fraction * 100.0
                    ),
                ))
            }
            _ => None,
        }
    }
}

fn generate_ising_nucleation_baseline() -> String {
    let mut tool = IsingModelTool {
        droplet_seeds: vec![DropletSeed {
            cx: 50,
            cy: 50,
            r: 2,
            spin: 1,
        }],
        brush_radius: 7,
        brush_cooperator: true,
        ..Default::default()
    };
    let _ = tool.set_ising_challenge(Some(CHALLENGE_ISING_DROPLET_NUCLEATION));
    tool.save_state().unwrap_or_default()
}

fn clamp_ising_nucleation_state(raw_json: &str) -> Result<String, String> {
    let mut tool = IsingModelTool::default();
    tool.state_bounds().validate_json(raw_json)?;
    let parsed: IsingState = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid IsingModelTool challenge JSON: {e}"))?;
    parsed.validate()?;
    tool.droplet_seeds = parsed.droplet_seeds;
    tool.brush_radius = parsed.brush_radius;
    tool.brush_cooperator = parsed.brush_cooperator;
    tool.set_ising_challenge(Some(CHALLENGE_ISING_DROPLET_NUCLEATION))?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped Ising challenge state".to_string())
}

fn generate_fermi_nucleation_baseline() -> String {
    let mut tool = IsingModelTool {
        droplet_seeds: vec![DropletSeed {
            cx: 50,
            cy: 50,
            r: 1,
            spin: 1,
        }],
        brush_radius: 6,
        brush_cooperator: true,
        ..Default::default()
    };
    let _ = tool.set_ising_challenge(Some(CHALLENGE_FERMI_PD_NUCLEATION));
    tool.save_state().unwrap_or_default()
}

fn clamp_fermi_nucleation_state(raw_json: &str) -> Result<String, String> {
    let mut tool = IsingModelTool::default();
    tool.state_bounds().validate_json(raw_json)?;
    let parsed: IsingState = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid IsingModelTool challenge JSON: {e}"))?;
    parsed.validate()?;
    tool.droplet_seeds = parsed.droplet_seeds;
    tool.brush_radius = parsed.brush_radius;
    tool.brush_cooperator = parsed.brush_cooperator;
    tool.set_ising_challenge(Some(CHALLENGE_FERMI_PD_NUCLEATION))?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped Fermi challenge state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_ISING_DROPLET_NUCLEATION,
        title: "Critical Droplet Nucleation (Becker-Döring)",
        description: "In a metastable -1 sea at T = 1.2, J = 1.0, h = 0.35, the baseline r = 2 droplet collapses under capillary line tension (Rc ≈ J/h ≈ 2.9). Paint a supercritical +1 droplet within ≤ 380 sites so magnetization nucleates past 6%.",
        category: "Statistical Mechanics",
        icon: "🫧",
        domain: "solid_state",
        tool_name: "IsingModelTool",
        locked_inputs: "Ferromagnetic, T = 1.2, J = 1.0, h = 0.35, 100×100 -1 sea",
        tunable_budget: "1..=4 circular droplets (r ∈ [0, 10], total +1 seed ≤ 380 sites)",
        target_summary: "Seeded +1 sites ∈ [1, 380] and Final +1 Density ≥ 6.0% after 45 sweeps",
        baseline_state_generator: generate_ising_nucleation_baseline,
        clamp_locked_state: clamp_ising_nucleation_state,
    }
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_FERMI_PD_NUCLEATION,
        title: "Prisoner's Dilemma Spatial Reciprocity",
        description: "Well-mixed 0D Replicator Dynamics drives Prisoner's Dilemma [R=1, S=-0.1, T=1.25, P=0] to 0% cooperation, and an isolated r = 1 seed dies. Nucleate a compact cooperator cluster (≤ 380 sites) so boundary reciprocity (2R + 2S > T) expands ≥ 8%.",
        category: "Evolutionary Spatial Games",
        icon: "🛡",
        domain: "solid_state",
        tool_name: "IsingModelTool",
        locked_inputs: "FermiGame Prisoner's Dilemma [1.0, -0.1, 1.25, 0.0], T = 0.08, μ = 0, -1 sea",
        tunable_budget: "1..=4 circular cooperator droplets (r ∈ [0, 10], total +1 seed ≤ 380 sites)",
        target_summary: "Seeded +1 sites ∈ [1, 380] and Spatial Cooperator Density ≥ 8.0% after 45 sweeps",
        baseline_state_generator: generate_fermi_nucleation_baseline,
        clamp_locked_state: clamp_fermi_nucleation_state,
    }
}
