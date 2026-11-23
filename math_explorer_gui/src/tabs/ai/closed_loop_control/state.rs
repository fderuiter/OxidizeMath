//! Serializable configuration, parameter bounds, and cross-domain starter presets for `ClosedLoopControlTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::ToolStateBounds;

/// Dynamical plant available in the closed-loop observer-controller workbench.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlantKind {
    /// Chaotic 3D `LorenzSystem` stabilized onto a Viswanath unstable periodic orbit via OGY Rayleigh thrusts.
    LorenzChaos,
    /// Compartmental `SEIRModel` reconstructing latent `Exposed` ($E$) from noisy infectious ($I$) counts.
    SeirEpidemic,
    /// Nonlinear ADME pharmacokinetics (`BatemanModel` / `SuperpositionModel`) holding plasma levels in `[MEC, MTC]`.
    Pharmacokinetics,
}

impl PlantKind {
    /// Human-readable plant title for UI selectors and accessibility descriptions.
    pub const fn label(self) -> &'static str {
        match self {
            Self::LorenzChaos => "Lorenz Chaos (OGY Orbit Lock)",
            Self::SeirEpidemic => "SEIR Outbreak (Latent E Throttling)",
            Self::Pharmacokinetics => "ADME Pharmacokinetics ([MEC, MTC])",
        }
    }

    /// Time-axis label for synchronized trajectories.
    pub const fn x_axis_label(self) -> &'static str {
        match self {
            Self::LorenzChaos => "Convection Time t",
            Self::SeirEpidemic => "Outbreak Time (days)",
            Self::Pharmacokinetics => "Clinical Time (hours)",
        }
    }

    /// Primary measured state label shown on side-by-side uncontrolled and closed-loop plots.
    pub const fn primary_y_label(self) -> &'static str {
        match self {
            Self::LorenzChaos => "Convection Mode x(t)",
            Self::SeirEpidemic => "Infectious Count I(t)",
            Self::Pharmacokinetics => "Plasma Concentration C_p (mg/L)",
        }
    }

    /// Reconstructed latent state label tracked by the Extended Kalman Filter.
    pub const fn latent_label(self) -> &'static str {
        match self {
            Self::LorenzChaos => "EKF Latent y(t)",
            Self::SeirEpidemic => "EKF Latent Exposed E(t)",
            Self::Pharmacokinetics => "EKF Clearance k_e (1/h) x10",
        }
    }

    /// Bounded control actuation channel label.
    pub const fn actuation_label(self) -> &'static str {
        match self {
            Self::LorenzChaos => "Rayleigh Thrust Δρ(t)",
            Self::SeirEpidemic => "Transmission Throttle u(t)",
            Self::Pharmacokinetics => "Adaptive Oral Dose D(t) (mg)",
        }
    }

    /// Fixed integration step $\Delta t$ for the selected dynamical plant.
    pub const fn dt(self) -> f64 {
        match self {
            Self::LorenzChaos => 1.558_652_210_716_2 / 156.0,
            Self::SeirEpidemic => 0.25,
            Self::Pharmacokinetics => 0.5,
        }
    }
}

/// Reinforcement learning controller family coupled to the EKF state estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ControllerKind {
    /// Critic-free Group Relative Policy Optimization (`normalize_group_advantages` + `clipped_surrogate_objective`).
    GrpoPolicy,
    /// Discretized tabular Q-learning (`QLearningAgent` + `FixedArrayQFunction` + `select_action_into`).
    DiscreteQLearning,
}

impl ControllerKind {
    /// Human-readable controller label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::GrpoPolicy => "GRPO Gaussian Policy",
            Self::DiscreteQLearning => "Discrete Q-Learning Agent",
        }
    }
}

/// Complete serializable state for `ClosedLoopControlTool` (`<= URL_JSON_BOUNDS`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosedLoopState {
    /// Dimensionless sensor noise scale (`[0.0, 5.0]`; floored in EKF covariance at `1e-8`).
    pub sensor_noise: f64,
    /// Dimensionless actuation authority multiplier (`[0.0, 6.0]`; `1.0` = nominal actuator limit).
    pub actuation_budget: f64,
    /// Selected nonlinear dynamical plant.
    pub plant: PlantKind,
    /// Selected RL controller architecture.
    pub controller: ControllerKind,
    /// EKF process noise variance scale $Q$ (`[1e-6, 2.0]`).
    pub process_noise: f64,
    /// GRPO proximal policy gradient learning rate (`[0.005, 0.80]`).
    pub grpo_lr: f64,
    /// GRPO surrogate clipping epsilon $\epsilon$ (`[0.05, 0.50]`).
    pub grpo_clip_eps: f64,
    /// GRPO KL divergence penalty coefficient $\beta_{\text{KL}}$ (`[0.0, 0.25]`).
    pub grpo_kl_beta: f64,
    /// `QLearningAgent` step-size $\alpha \in [0.01, 0.95]$.
    pub q_alpha: f64,
    /// `QLearningAgent` discount factor $\gamma \in [0.50, 0.995]$.
    pub q_gamma: f64,
    /// `EpsilonGreedy` exploration probability $\epsilon \in [0.0, 0.80]$.
    pub q_epsilon: f64,
    /// Normalized perturbation impulse applied by click-to-kick interactions (`[0.05, 3.0]`).
    pub kick_magnitude: f64,
    /// Pharmacokinetic Minimum Effective Concentration in `mg/L` (`[0.2, 5.0]`).
    pub pk_mec: f64,
    /// Pharmacokinetic Maximum Tolerated Concentration in `mg/L` (`[2.0, 15.0]`, `> pk_mec`).
    pub pk_mtc: f64,
    /// Warm-start RL training episodes executed on rebuild (`[0, 20]`).
    pub warmup_episodes: usize,
    /// Deterministic `OxidizeRng` seed for sensor noise, policy sampling, and perturbations.
    pub rng_seed: u64,
}

impl Default for ClosedLoopState {
    fn default() -> Self {
        Self {
            sensor_noise: 0.25,
            actuation_budget: 1.15,
            plant: PlantKind::LorenzChaos,
            controller: ControllerKind::GrpoPolicy,
            process_noise: 0.02,
            grpo_lr: 0.22,
            grpo_clip_eps: 0.20,
            grpo_kl_beta: 0.02,
            q_alpha: 0.28,
            q_gamma: 0.92,
            q_epsilon: 0.12,
            kick_magnitude: 1.0,
            pk_mec: 1.2,
            pk_mtc: 3.2,
            warmup_episodes: 6,
            rng_seed: 42,
        }
    }
}

impl ClosedLoopState {
    /// Strict `ToolStateBounds` well within `URL_JSON_BOUNDS` (`4096` bytes).
    pub const STATE_BOUNDS: ToolStateBounds = ToolStateBounds::new(2048, 4, 32);

    /// Checks that all floating-point parameters are finite IEEE-754 numbers.
    pub fn is_finite(&self) -> bool {
        [
            self.sensor_noise,
            self.actuation_budget,
            self.process_noise,
            self.grpo_lr,
            self.grpo_clip_eps,
            self.grpo_kl_beta,
            self.q_alpha,
            self.q_gamma,
            self.q_epsilon,
            self.kick_magnitude,
            self.pk_mec,
            self.pk_mtc,
        ]
        .iter()
        .all(|v| v.is_finite())
    }

    /// Validates all physical, RL, and therapeutic window bounds.
    pub fn validate_bounds(&self) -> Result<(), String> {
        check_range("sensor_noise", self.sensor_noise, 0.0, 5.0)?;
        check_range("actuation_budget", self.actuation_budget, 0.0, 6.0)?;
        check_range("process_noise", self.process_noise, 1.0e-6, 2.0)?;
        check_range("grpo_lr", self.grpo_lr, 0.005, 0.80)?;
        check_range("grpo_clip_eps", self.grpo_clip_eps, 0.05, 0.50)?;
        check_range("grpo_kl_beta", self.grpo_kl_beta, 0.0, 0.25)?;
        check_range("q_alpha", self.q_alpha, 0.01, 0.95)?;
        check_range("q_gamma", self.q_gamma, 0.50, 0.995)?;
        check_range("q_epsilon", self.q_epsilon, 0.0, 0.80)?;
        check_range("kick_magnitude", self.kick_magnitude, 0.05, 3.0)?;
        check_range("pk_mec", self.pk_mec, 0.2, 5.0)?;
        check_range("pk_mtc", self.pk_mtc, 2.0, 15.0)?;
        if self.pk_mtc <= self.pk_mec + 0.25 {
            return Err(format!(
                "pk_mtc ({}) must exceed pk_mec ({}) by at least 0.25 mg/L",
                self.pk_mtc, self.pk_mec
            ));
        }
        if self.warmup_episodes > 20 {
            return Err(format!(
                "warmup_episodes out of [0, 20]: {}",
                self.warmup_episodes
            ));
        }
        Ok(())
    }

    /// Clamps potentially out-of-range or non-finite parameters into safe validated intervals.
    pub fn clamp_in_place(&mut self) {
        self.sensor_noise = sanitize_clamp(self.sensor_noise, 0.25, 0.0, 5.0);
        self.actuation_budget = sanitize_clamp(self.actuation_budget, 1.15, 0.0, 6.0);
        self.process_noise = sanitize_clamp(self.process_noise, 0.02, 1.0e-6, 2.0);
        self.grpo_lr = sanitize_clamp(self.grpo_lr, 0.22, 0.005, 0.80);
        self.grpo_clip_eps = sanitize_clamp(self.grpo_clip_eps, 0.20, 0.05, 0.50);
        self.grpo_kl_beta = sanitize_clamp(self.grpo_kl_beta, 0.02, 0.0, 0.25);
        self.q_alpha = sanitize_clamp(self.q_alpha, 0.28, 0.01, 0.95);
        self.q_gamma = sanitize_clamp(self.q_gamma, 0.92, 0.50, 0.995);
        self.q_epsilon = sanitize_clamp(self.q_epsilon, 0.12, 0.0, 0.80);
        self.kick_magnitude = sanitize_clamp(self.kick_magnitude, 1.0, 0.05, 3.0);
        self.pk_mec = sanitize_clamp(self.pk_mec, 1.2, 0.2, 4.5);
        let min_mtc = (self.pk_mec + 0.5).max(2.0);
        self.pk_mtc = sanitize_clamp(self.pk_mtc, 3.2, min_mtc, 15.0);
        self.warmup_episodes = self.warmup_episodes.min(20);
    }
}

fn check_range(name: &str, val: f64, min: f64, max: f64) -> Result<(), String> {
    if !val.is_finite() || !(min..=max).contains(&val) {
        return Err(format!("{name} out of [{min}, {max}]: {val}"));
    }
    Ok(())
}

fn sanitize_clamp(val: f64, fallback: f64, min: f64, max: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback.clamp(min, max)
    }
}

/// Starter preset pairing `ClosedLoopControlTool` with `ChaosTab` for Lorenz OGY stabilization.
pub const STARTER_LORENZ_OGY_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lorenz OGY Periodic Orbit Stabilization (EKF + GRPO)",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "sensor_noise": 0.30,
    "actuation_budget": 1.25,
    "plant": "LorenzChaos",
    "controller": "GrpoPolicy",
    "process_noise": 0.02,
    "grpo_lr": 0.24,
    "grpo_clip_eps": 0.20,
    "grpo_kl_beta": 0.02,
    "q_alpha": 0.28,
    "q_gamma": 0.92,
    "q_epsilon": 0.12,
    "kick_magnitude": 1.0,
    "pk_mec": 1.2,
    "pk_mtc": 3.2,
    "warmup_episodes": 8,
    "rng_seed": 2026
  }
}"#;

/// Starter preset pairing `ClosedLoopControlTool` with `EpidemiologyTab` for SEIR outbreak suppression.
pub const STARTER_SEIR_OUTBREAK_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "SEIR Latent-Exposed EKF + GRPO Outbreak Suppression",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "sensor_noise": 0.22,
    "actuation_budget": 1.10,
    "plant": "SeirEpidemic",
    "controller": "GrpoPolicy",
    "process_noise": 0.04,
    "grpo_lr": 0.25,
    "grpo_clip_eps": 0.20,
    "grpo_kl_beta": 0.02,
    "q_alpha": 0.30,
    "q_gamma": 0.94,
    "q_epsilon": 0.10,
    "kick_magnitude": 1.1,
    "pk_mec": 1.2,
    "pk_mtc": 3.2,
    "warmup_episodes": 8,
    "rng_seed": 2027
  }
}"#;

/// Starter preset pairing `ClosedLoopControlTool` with `ClinicalTrialsTab` for `[MEC, MTC]` dosing.
pub const STARTER_PK_DOSING_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Adaptive Bateman PK Dosing in [MEC, MTC] (EKF + Q-Learning)",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "sensor_noise": 0.12,
    "actuation_budget": 1.15,
    "plant": "Pharmacokinetics",
    "controller": "DiscreteQLearning",
    "process_noise": 0.015,
    "grpo_lr": 0.22,
    "grpo_clip_eps": 0.20,
    "grpo_kl_beta": 0.02,
    "q_alpha": 0.35,
    "q_gamma": 0.90,
    "q_epsilon": 0.10,
    "kick_magnitude": 0.9,
    "pk_mec": 1.2,
    "pk_mtc": 3.2,
    "warmup_episodes": 12,
    "rng_seed": 2028
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "closed_loop_lorenz_ogy_starter",
        title: "Lorenz OGY Periodic Orbit Stabilization (EKF + GRPO)",
        description: "Reconstructs hidden 3D Lorenz state from noisy scalar x(t) and stabilizes an unstable periodic orbit via bounded OGY Rayleigh thrusts; pairs in split-pane with ChaosTab.",
        category: "AI & Control",
        icon: "🦋",
        domain: "ai",
        tool_name: Some("ClosedLoopControlTool"),
        preset_generator: || STARTER_LORENZ_OGY_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "closed_loop_seir_outbreak_starter",
        title: "SEIR Latent-Exposed EKF + GRPO Outbreak Suppression",
        description: "Infers latent Exposed E(t) burden from noisy infectious counts I(t) and throttles transmission beta_eff(t) against an uncontrolled epidemic twin; pairs with EpidemiologyTab.",
        category: "AI & Control",
        icon: "🦠",
        domain: "ai",
        tool_name: Some("ClosedLoopControlTool"),
        preset_generator: || STARTER_SEIR_OUTBREAK_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "closed_loop_pk_dosing_starter",
        title: "Adaptive Bateman PK Dosing in [MEC, MTC] (EKF + Q-Learning)",
        description: "Estimates gut absorption mass A_g and patient clearance k_e from noisy plasma assays to hold drug concentration inside [MEC, MTC] vs SuperpositionModel; pairs with ClinicalTrialsTab.",
        category: "AI & Control",
        icon: "💊",
        domain: "ai",
        tool_name: Some("ClosedLoopControlTool"),
        preset_generator: || STARTER_PK_DOSING_JSON,
    }
}

// [cite:grpo]
