//! Federated `mri_kspace_radial_cs` inverse challenge and `CalloutManager` sync for [`MriKSpaceTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_runner::MriKSpaceRunner;
use super::kspace_tool::MriKSpaceTool;
use super::kspace_types::{
    KSpaceMaskPreset, MriKSpaceSnapshot, MriKSpaceToolState, PhantomPreset, StudioBrushMode,
};
use crate::async_sim::{SimWorkerState, SimulationRunner};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};
use std::sync::Arc;

/// Canonical challenge ID for 2D radial compressed-sensing MRI reconstruction on Shepp-Logan tissue.
pub const CHALLENGE_MRI_KSPACE_RADIAL_CS: &str = "mri_kspace_radial_cs";

/// Locked RF flip angle in degrees for proton-density-weighted Shepp-Logan readout.
pub const CHALLENGE_MRI_FLIP_DEG: f64 = 90.0;
/// Locked repetition time $T_R$ in seconds (`3.5 s`) minimizing $T_1$ saturation bias against $\rho$.
pub const CHALLENGE_MRI_TR: f64 = 3.5;
/// Locked echo time $T_E$ in seconds (`0.008 s`) minimizing $T_2$ decay bias against $\rho$.
pub const CHALLENGE_MRI_TE: f64 = 0.008;
/// Locked spatial/k-space matrix size (`64 x 64`).
pub const CHALLENGE_MRI_GRID_SIZE: usize = 64;
/// Minimum required k-space scan acceleration factor $R = N^2 / N_{\text{sampled}}$.
pub const MIN_ACCELERATION_R: f64 = 2.5;
/// Maximum allowed normalized root-mean-square error against $\rho$ in percent.
pub const MAX_NRMSE_PCT: f64 = 48.0;

impl MriKSpaceTool {
    /// Returns `true` when `mri_kspace_radial_cs` locks phantom and RF contrast parameters.
    #[must_use]
    pub(crate) fn is_challenge_locked(&self) -> bool {
        self.active_challenge == Some(CHALLENGE_MRI_KSPACE_RADIAL_CS)
    }

    /// Restores locked Shepp-Logan phantom, proton-density RF sequence, and k-space-only strokes.
    pub(crate) fn enforce_challenge_locks_on_state(cfg: &mut MriKSpaceToolState) {
        cfg.phantom = PhantomPreset::SheppLogan;
        cfg.grid_size = CHALLENGE_MRI_GRID_SIZE;
        cfg.flip_angle_deg = CHALLENGE_MRI_FLIP_DEG;
        cfg.tr = CHALLENGE_MRI_TR;
        cfg.te = CHALLENGE_MRI_TE;
        cfg.grad_gx_integral = 0.0;
        cfg.grad_gy_integral = 0.0;
        cfg.evolve_turing = false;
        cfg.selected_voxel[0] = cfg.selected_voxel[0].min(CHALLENGE_MRI_GRID_SIZE - 1);
        cfg.selected_voxel[1] = cfg.selected_voxel[1].min(CHALLENGE_MRI_GRID_SIZE - 1);
        if !matches!(
            cfg.brush_mode,
            StudioBrushMode::KSpaceMaskErase | StudioBrushMode::KSpaceMaskPaint
        ) {
            cfg.brush_mode = StudioBrushMode::KSpaceMaskErase;
        }
        cfg.strokes.retain(|s| {
            matches!(
                StudioBrushMode::from_code(s.0[0]),
                Some(StudioBrushMode::KSpaceMaskErase | StudioBrushMode::KSpaceMaskPaint)
            )
        });
    }

    /// Rebuilds the worker snapshot immediately after challenge activation or lock enforcement.
    pub(crate) fn rebuild_runner_snapshot(&mut self) {
        if self.is_challenge_locked() {
            Self::enforce_challenge_locks_on_state(&mut self.config);
        }
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .config = self.config.clone();
        let fresh = MriKSpaceRunner::new(Arc::clone(&self.params));
        let fresh_snap = fresh.get_snapshot();
        self.latest_sonification = fresh_snap.sonification;
        if let Some(snap) = fresh_snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MriKSpaceSnapshot>())
        {
            self.latest = snap.clone();
        }
        self.fault_notified = false;
        let params = Arc::clone(&self.params);
        self.worker
            .reset_or_respawn(move || MriKSpaceRunner::new(params));
        self.textures_dirty = true;
    }

    /// Activates or clears `mri_kspace_radial_cs` on [`MriKSpaceTool`].
    pub(crate) fn set_kspace_challenge(
        &mut self,
        challenge_id: Option<&str>,
    ) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                Ok(())
            }
            Some(CHALLENGE_MRI_KSPACE_RADIAL_CS) => {
                self.active_challenge = Some(CHALLENGE_MRI_KSPACE_RADIAL_CS);
                self.rebuild_runner_snapshot();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for MriKSpaceStudio"
            )),
        }
    }

    /// Evaluates `MriKSpaceSnapshot::{nrmse_pct, acceleration_r}` against `R >= 2.5` and `NRMSE <= 48%`.
    #[must_use]
    pub(crate) fn eval_kspace_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge?;
        let r = self.latest.acceleration_r;
        let nrmse = self.latest.nrmse_pct;
        Some(score_kspace_cs(r, nrmse))
    }

    /// Synchronizes `CalloutManager` markers onto the 3D Bloch readout plot.
    pub(crate) fn sync_kspace_callouts(&mut self) {
        let net = self.latest.selected_net_m;
        let tip = self.config.camera.project(&[net.x, net.y, net.z]);
        self.callout_manager.set_auto_callout(
            "mri_kspace_metrics",
            [-0.95, 1.05],
            format!(
                "NRMSE = {:.1}% | R = {:.2}x",
                self.latest.nrmse_pct, self.latest.acceleration_r
            ),
            [0.0_f32, -14.0_f32],
        );
        self.callout_manager.set_auto_callout(
            "mri_kspace_net_m",
            tip,
            format!("|M_xy| = {:.2}", net.xy().norm()),
            [18.0_f32, -16.0_f32],
        );
    }
}

/// Pure scorer evaluating compressed-sensing acceleration $R$ and reconstruction NRMSE (%).
#[must_use]
pub fn score_kspace_cs(acceleration_r: f64, nrmse_pct: f64) -> ChallengeEvaluation {
    let r_ok = acceleration_r.is_finite() && acceleration_r >= MIN_ACCELERATION_R;
    let err_ok = nrmse_pct.is_finite() && nrmse_pct <= MAX_NRMSE_PCT;
    let passed = r_ok && err_ok;

    let r_deficit = (MIN_ACCELERATION_R - acceleration_r).max(0.0);
    let err_excess = (nrmse_pct - MAX_NRMSE_PCT).max(0.0);
    let residual = err_excess + 20.0 * r_deficit;
    let score = if passed {
        let bonus = ((MAX_NRMSE_PCT - nrmse_pct) / MAX_NRMSE_PCT).clamp(0.0, 1.0) * 10.0;
        (90.0 + bonus).clamp(90.0, 100.0)
    } else {
        let r_term = (acceleration_r / MIN_ACCELERATION_R).clamp(0.0, 1.0) * 45.0;
        let e_term = (1.0 - (err_excess / 52.0).clamp(0.0, 1.0)) * 44.0;
        (r_term + e_term).clamp(0.0, 89.9)
    };
    ChallengeEvaluation::new(
        score,
        passed,
        "NRMSE (%) / R deficit",
        if passed { nrmse_pct } else { residual },
        format!(
            "Acceleration R = {acceleration_r:.2}x (need >= {MIN_ACCELERATION_R:.1}x), NRMSE = {nrmse_pct:.1}% (need <= {MAX_NRMSE_PCT:.1}%)"
        ),
    )
}

fn generate_kspace_cs_baseline() -> String {
    let mut cfg = MriKSpaceToolState {
        flip_angle_deg: CHALLENGE_MRI_FLIP_DEG,
        tr: CHALLENGE_MRI_TR,
        te: CHALLENGE_MRI_TE,
        mask_cutoff: 0.30,
        grad_gx_integral: 0.0,
        grad_gy_integral: 0.0,
        grid_size: CHALLENGE_MRI_GRID_SIZE,
        radial_spokes: 16,
        cartesian_stride: 2,
        brush_radius: 4,
        selected_voxel: [32, 32],
        phantom: PhantomPreset::SheppLogan,
        mask_preset: KSpaceMaskPreset::Full,
        brush_mode: StudioBrushMode::KSpaceMaskErase,
        evolve_turing: false,
        state: Some(SimWorkerState::Ready),
        ..MriKSpaceToolState::default()
    };
    MriKSpaceTool::enforce_challenge_locks_on_state(&mut cfg);
    serde_json::to_string(&cfg).unwrap_or_default()
}

fn clamp_kspace_cs_state(raw_json: &str) -> Result<String, String> {
    let mut val: serde_json::Value = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid MriKSpaceStudio challenge JSON: {e}"))?;
    let obj = val
        .as_object_mut()
        .ok_or_else(|| "MriKSpaceStudio challenge state must be a JSON object".to_string())?;
    obj.insert("phantom".to_string(), serde_json::json!("shepp_logan"));
    obj.insert(
        "grid_size".to_string(),
        serde_json::json!(CHALLENGE_MRI_GRID_SIZE),
    );
    obj.insert(
        "flip_angle_deg".to_string(),
        serde_json::json!(CHALLENGE_MRI_FLIP_DEG),
    );
    obj.insert("tr".to_string(), serde_json::json!(CHALLENGE_MRI_TR));
    obj.insert("te".to_string(), serde_json::json!(CHALLENGE_MRI_TE));
    obj.insert("grad_gx_integral".to_string(), serde_json::json!(0.0));
    obj.insert("grad_gy_integral".to_string(), serde_json::json!(0.0));
    obj.insert("evolve_turing".to_string(), serde_json::json!(false));
    if let Some(spokes) = obj.get("radial_spokes").and_then(serde_json::Value::as_u64) {
        obj.insert(
            "radial_spokes".to_string(),
            serde_json::json!((spokes as usize).clamp(4, 64)),
        );
    }
    if let Some(stride) = obj
        .get("cartesian_stride")
        .and_then(serde_json::Value::as_u64)
    {
        obj.insert(
            "cartesian_stride".to_string(),
            serde_json::json!((stride as usize).clamp(1, 8)),
        );
    }
    if let Some(cutoff) = obj.get("mask_cutoff").and_then(serde_json::Value::as_f64) {
        let safe = if cutoff.is_finite() {
            cutoff.clamp(0.05, 1.0)
        } else {
            0.30
        };
        obj.insert("mask_cutoff".to_string(), serde_json::json!(safe));
    }

    let sanitized = serde_json::to_string(&val).map_err(|e| e.to_string())?;
    let mut tool = MriKSpaceTool::default();
    tool.set_kspace_challenge(Some(CHALLENGE_MRI_KSPACE_RADIAL_CS))?;
    tool.try_load_state(&sanitized)?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped MriKSpaceStudio state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_MRI_KSPACE_RADIAL_CS,
        title: "Radial k-Space Compressed Sensing (R >= 2.5x)",
        description: "Undersample 2D Shepp-Logan k-space using radial spokes or masks to achieve scan acceleration R >= 2.5x while keeping reconstruction NRMSE <= 48%.",
        category: "Medical Physics",
        icon: "🧠",
        domain: "mri",
        tool_name: "MriKSpaceStudio",
        locked_inputs: "SheppLogan 64×64, α = 90°, TR = 3.5 s, TE = 8 ms, ΔB0/RF spatial strokes disabled",
        tunable_budget: "Mask preset, radial spokes ∈ [4, 64], cutoff ∈ [0.05, 1.0], stride ∈ [1, 8], k-space mask brush",
        target_summary: "Acceleration R >= 2.5x and reconstruction NRMSE <= 48.0% vs. ρ",
        baseline_state_generator: generate_kspace_cs_baseline,
        clamp_locked_state: clamp_kspace_cs_state,
    }
}

// [cite:graph_parameters_rust]
