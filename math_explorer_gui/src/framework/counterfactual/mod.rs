//! Counterfactual shadow twin (`CounterfactualBranchEngine`) framework primitives.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod metrics;
mod ui;

#[cfg(test)]
mod tests;

pub use metrics::*;
pub use ui::*;

/// Minimum non-zero state perturbation magnitude (`1e-8`).
pub const MIN_PERTURBATION_DELTA: f64 = 1e-8;

/// Maximum state perturbation magnitude (`1.0`).
pub const MAX_PERTURBATION_DELTA: f64 = 1.0;

/// Default state perturbation magnitude (`1e-5`, classic chaotic butterfly nudge).
pub const DEFAULT_PERTURBATION_DELTA: f64 = 1e-5;

/// Clamps positive state perturbations to `[1e-8, 1.0]` while preserving `0.0` for exact parity runs.
#[must_use]
pub fn sanitize_perturbation_delta(delta: f64) -> f64 {
    if !delta.is_finite() || delta <= 0.0 {
        0.0
    } else {
        delta.clamp(MIN_PERTURBATION_DELTA, MAX_PERTURBATION_DELTA)
    }
}

/// Domain-specific counterfactual intervention delta displayed on the Divergence Strip.
#[derive(Clone, Debug, PartialEq)]
pub struct DomainInterventionDelta {
    /// Short metric label (e.g., `"ΔI_peak"`, `"ΔM_xy"`, `"||u_B - u_A||_∞"`).
    pub label: &'static str,
    /// Formatted value string for display and clipboard copy.
    pub formatted: String,
    /// Underlying finite floating-point scalar value.
    pub raw_value: f64,
    /// Accessible hover explanation of the domain metric.
    pub tooltip: &'static str,
}

impl DomainInterventionDelta {
    /// Constructs a new domain intervention delta entry.
    #[must_use]
    pub fn new(
        label: &'static str,
        formatted: String,
        raw_value: f64,
        tooltip: &'static str,
    ) -> Self {
        Self {
            label,
            formatted,
            raw_value: if raw_value.is_finite() {
                raw_value
            } else {
                0.0
            },
            tooltip,
        }
    }
}

/// Live snapshot of divergence metrics between `Branch A` (baseline) and `Branch B` (active).
#[derive(Clone, Debug, PartialEq)]
pub struct DivergenceSnapshot {
    /// Discrete step index `k` where the fork was anchored.
    pub fork_step: usize,
    /// Simulation time `t_k` where the fork was anchored.
    pub fork_time: f64,
    /// Current simulation time `t >= t_k`.
    pub current_time: f64,
    /// Applied state perturbation magnitude `δx` (`0.0` or `1e-8..=1.0`).
    pub delta_x: f64,
    /// Instantaneous Euclidean state/field separation `||x_B(t) - x_A(t)||_2`.
    pub l2_separation: f64,
    /// Finite-time logarithmic separation slope `ln(||δx(t)|| / ||δx_0||) / Δt`.
    pub log_separation_slope: f64,
    /// Maximum separation `max_{τ ∈ [t_k, t]} ||x_B(τ) - x_A(τ)||_2`.
    pub peak_separation: f64,
    /// Primary domain intervention delta (e.g., `ΔI_peak` or `ΔM_xy`).
    pub primary_delta: Option<DomainInterventionDelta>,
    /// Secondary domain intervention delta (e.g., spatial `L_∞` norm).
    pub secondary_delta: Option<DomainInterventionDelta>,
}

impl DivergenceSnapshot {
    /// Formats a single-line plain-text summary suitable for clipboard export.
    #[must_use]
    pub fn summary_clipboard_text(&self, tool_name: &str) -> String {
        let mut out = format!(
            "{tool_name} | Fork @ t_k={:.3} (step {}) | t={:.3} | δx={:.2e} | ||x_B-x_A||_2={:.4e} | ln||δx||/Δt={:+.4} | peak={:.4e}",
            self.fork_time,
            self.fork_step,
            self.current_time,
            self.delta_x,
            self.l2_separation,
            self.log_separation_slope,
            self.peak_separation,
        );
        if let Some(ref d1) = self.primary_delta {
            out.push_str(&format!(" | {}={}", d1.label, d1.formatted));
        }
        if let Some(ref d2) = self.secondary_delta {
            out.push_str(&format!(" | {}={}", d2.label, d2.formatted));
        }
        out
    }
}

/// User actions emitted by `CounterfactualBranchEngine` transport controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BranchControlAction {
    /// Clone `(t_k, x(t_k), rng)` into shadow twin `Branch A` and apply perturbation to `Branch B`.
    ForkHere,
    /// Swap `Branch A` (baseline) and `Branch B` (active) trajectories and parameters.
    SwapBranches,
    /// Discard shadow twin `Branch A` and keep `Branch B`.
    ClearFork,
    /// Update configured state perturbation `δx` (`0.0` or `1e-8..=1.0`).
    SetPerturbation(f64),
}

/// Framework-level state for mid-flight counterfactual twin branching and divergence readouts.
#[derive(Clone, Debug)]
pub struct CounterfactualBranchEngine {
    /// Non-zero magnitude selected on the logarithmic slider (`1e-8..=1.0`).
    slider_delta: f64,
    /// Whether state perturbation is enabled (`false` yields `δx = 0.0` parameter-only / parity fork).
    perturbation_enabled: bool,
    /// Latest divergence snapshot synchronized from the active pilot tool.
    pub active_snapshot: Option<DivergenceSnapshot>,
}

impl Default for CounterfactualBranchEngine {
    fn default() -> Self {
        Self {
            slider_delta: DEFAULT_PERTURBATION_DELTA,
            perturbation_enabled: true,
            active_snapshot: None,
        }
    }
}

impl CounterfactualBranchEngine {
    /// Constructs a new `CounterfactualBranchEngine` with a specific initial `δx`.
    #[must_use]
    pub fn with_delta(delta: f64) -> Self {
        let mut engine = Self::default();
        engine.set_effective_delta(delta);
        engine
    }

    /// Effective state perturbation `δx` (`0.0` when disabled, otherwise in `1e-8..=1.0`).
    #[must_use]
    pub fn effective_delta(&self) -> f64 {
        if self.perturbation_enabled {
            sanitize_perturbation_delta(self.slider_delta)
        } else {
            0.0
        }
    }

    /// Updates `δx` (`0.0` disables state perturbation; positive values clamp to `1e-8..=1.0`).
    pub fn set_effective_delta(&mut self, delta: f64) {
        let clean = sanitize_perturbation_delta(delta);
        if clean == 0.0 {
            self.perturbation_enabled = false;
        } else {
            self.perturbation_enabled = true;
            self.slider_delta = clean;
        }
    }

    /// Returns `true` when an active counterfactual twin fork is present.
    #[must_use]
    pub const fn is_forked(&self) -> bool {
        self.active_snapshot.is_some()
    }

    /// Synchronizes the framework's cached divergence snapshot with the active tool.
    pub fn sync_snapshot(&mut self, snapshot: Option<DivergenceSnapshot>) {
        if let Some(ref snap) = snapshot {
            self.set_effective_delta(snap.delta_x);
        }
        self.active_snapshot = snapshot;
    }

    /// Clears the active divergence snapshot (e.g., on `Reset` or `FiniteStateGuard` rollback).
    pub fn clear(&mut self) {
        self.active_snapshot = None;
    }
}
