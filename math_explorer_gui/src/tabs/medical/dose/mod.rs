//! Interactive radiation therapy dose calculation, 4-beam IMRT inverse planning,
//! RLE contour sculpting, and live 4D radar-gated delivery workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod canvas;
#[cfg(test)]
mod tests;
pub mod ui;

use super::imrt::motion::{
    linac_energy_and_fluence_scale, Standalone4dStepper, NOMINAL_BEAM_CURRENT_MA,
};
use super::imrt::{
    DoseState, ImrtWorkspace, BASELINE_IMRT_ANGLES_DEG, BASELINE_IMRT_WEIGHTS, CHALLENGE_IMRT_BETA,
    CHALLENGE_IMRT_OAR_LIMIT_GY, CHALLENGE_IMRT_OAR_SPARING, CHALLENGE_IMRT_PRESCRIPTION_GY,
    IMRT_BEAM_COUNT, IMRT_GRID_DIM, OAR_SPARING_4D_ANGLES_DEG, OAR_SPARING_4D_WEIGHTS,
};
use super::link_bus::{current_write_seq, drain_gating_telemetry_into, GatingTelemetry};
use crate::framework::{
    BrushMode, CalloutManager, ChallengeEvaluation, ExportCapabilities, InteractionContext,
    InteractiveTool, SpatialBrushKit, ToolStateBounds,
};
use eframe::egui::{self, ColorImage};
use math_commons::math_kernel::marching_squares::ScalarField2D;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};
use math_explorer::physics::medical::radar_gating::BeamState;

pub use super::imrt::state::{CanvasKeyTarget, DoseDisplayMode};

/// Interactive radiation therapy tool supporting point-source PSF convolution, 4-beam IMRT
/// inverse planning, `SpatialBrushKit` contour sculpting, and live 4D radar-gated dosimetry.
pub struct DoseCalculationTool {
    pub(crate) amplitude: f64,
    pub(crate) beta: f64,
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Whether 4-beam coplanar IMRT mode is active.
    pub imrt_mode: bool,
    /// Whether 4D respiratory-gated delivery accumulation is active.
    pub gating_4d_enabled: bool,
    /// Whether single-pane/coupled 4D continuous playback is active.
    pub playing_4d: bool,
    /// Active dose matrix displayed on the heatmap (`Static`, `Gated 4D`, or `Ungated 4D`).
    pub display_mode: DoseDisplayMode,
    /// LINAC pulse gun current `I_b` (mA) scaling fluence via `accelerator::beam_loading_energy`.
    pub beam_current_ma: f64,
    /// Coplanar gantry beam angles (degrees) in `[0, 360]`.
    pub beam_angles_deg: [f64; IMRT_BEAM_COUNT],
    /// Coplanar beam weights in `[0, 10]`.
    pub beam_weights: [f64; IMRT_BEAM_COUNT],
    /// Target tumor prescription `D_Rx` (Gy).
    pub prescription_gy: f64,
    /// Organ-at-risk maximum dose limit `D_lim` (Gy).
    pub oar_limit_gy: f64,
    /// Active inverse challenge ID, if any.
    pub active_challenge: Option<&'static str>,
    /// Preallocated zero-allocation 4-beam IMRT and 4D motion workspace.
    pub imrt_workspace: ImrtWorkspace,
    /// Automated and manual plot callouts.
    pub callout_manager: CalloutManager,
    /// Reusable spatial brush controller for PTV/OAR contour sculpting.
    pub brush_kit: SpatialBrushKit,
    /// Active keyboard arrow-key target when the phantom plot holds focus.
    pub canvas_key_target: CanvasKeyTarget,
    /// Selected coplanar beam index (`0..4`) for keyboard gantry nudging.
    pub selected_beam: usize,
    /// Coplanar beam handle (`0..4`) currently dragged by pointer input.
    pub dragging_beam: Option<usize>,
    /// Plot widget ID when the phantom canvas holds keyboard focus.
    pub focused_canvas: Option<egui::Id>,
    /// Monotonic read cursor on the thread-local medical `link_bus`.
    pub bus_cursor_seq: u64,
    /// Whether the most recent 4D step arrived from `RadarGatingTool` over `link_bus`.
    pub linked_to_radar_bus: bool,
    /// Latest `GatingTelemetry` snapshot ingested from `link_bus` or `standalone_4d`.
    pub latest_telemetry: GatingTelemetry,
    /// Embedded single-pane 4D respiratory Kalman + gating stepper.
    pub standalone_4d: Standalone4dStepper,
    pub(crate) high_contrast: bool,
    pub(crate) texture: Option<egui::TextureHandle>,
}

impl Default for DoseCalculationTool {
    fn default() -> Self {
        let standalone_4d = Standalone4dStepper::default();
        let latest_telemetry = standalone_4d.latest;
        let mut tool = Self {
            amplitude: 1.0,
            beta: 1.0,
            width: 200,
            height: 200,
            imrt_mode: false,
            gating_4d_enabled: false,
            playing_4d: false,
            display_mode: DoseDisplayMode::StaticPlan,
            beam_current_ma: NOMINAL_BEAM_CURRENT_MA,
            beam_angles_deg: BASELINE_IMRT_ANGLES_DEG,
            beam_weights: BASELINE_IMRT_WEIGHTS,
            prescription_gy: CHALLENGE_IMRT_PRESCRIPTION_GY,
            oar_limit_gy: CHALLENGE_IMRT_OAR_LIMIT_GY,
            active_challenge: None,
            imrt_workspace: ImrtWorkspace::default(),
            callout_manager: CalloutManager::new(),
            brush_kit: SpatialBrushKit::new(BrushMode::PtvTarget, 0.95)
                .with_radius_bounds(0.35, 3.85, 0.35)
                .with_cursor_step(0.50, 0.50),
            canvas_key_target: CanvasKeyTarget::Brush,
            selected_beam: 0,
            dragging_beam: None,
            focused_canvas: None,
            bus_cursor_seq: current_write_seq(),
            linked_to_radar_bus: false,
            latest_telemetry,
            standalone_4d,
            high_contrast: false,
            texture: None,
        };
        tool.refresh_model();
        tool
    }
}

impl DoseCalculationTool {
    /// Re-evaluates the IMRT workspace (when active), updates callouts, and invalidates the texture.
    pub fn refresh_model(&mut self) {
        if self.active_challenge == Some(CHALLENGE_IMRT_OAR_SPARING) {
            self.imrt_mode = true;
            self.gating_4d_enabled = false;
            self.playing_4d = false;
            self.display_mode = DoseDisplayMode::StaticPlan;
            self.beta = CHALLENGE_IMRT_BETA;
            self.width = IMRT_GRID_DIM;
            self.height = IMRT_GRID_DIM;
            self.prescription_gy = CHALLENGE_IMRT_PRESCRIPTION_GY;
            self.oar_limit_gy = CHALLENGE_IMRT_OAR_LIMIT_GY;
            self.imrt_workspace.reset_contours_to_canonical();
        }
        let (e_mev, scale) = linac_energy_and_fluence_scale(self.beam_current_ma);
        self.imrt_workspace.motion.beam_energy_mev = e_mev;
        self.imrt_workspace.motion.fluence_scale = scale;

        if self.imrt_mode {
            self.imrt_workspace.evaluate_in_place(
                &self.beam_angles_deg,
                &self.beam_weights,
                self.beta,
                self.prescription_gy,
                self.oar_limit_gy,
            );
            self.imrt_workspace.sync_callouts(
                self.prescription_gy,
                self.oar_limit_gy,
                &mut self.callout_manager,
            );
        } else {
            let d_ref = ExponentialKernel::new(self.amplitude, self.beta)
                .value_at(2.0)
                .unwrap_or(0.0);
            let src_lbl = format!("Source A={:.1}, β={:.2}", self.amplitude, self.beta);
            self.callout_manager
                .set_auto_callout("psf_source", [0.0, 0.0], src_lbl, [28.0, -22.0]);
            let ref_lbl = format!("K(2cm) = {:.3}", d_ref);
            self.callout_manager
                .set_auto_callout("psf_r2", [2.0, 0.0], ref_lbl, [28.0, 20.0]);
        }
        self.texture = None;
    }

    /// Loads an OAR-sparing 4-beam coplanar plan ready for 4D respiratory-gated delivery.
    pub fn apply_oar_sparing_4d_preset(&mut self) {
        if self.active_challenge.is_some() {
            return;
        }
        self.imrt_mode = true;
        self.gating_4d_enabled = true;
        self.display_mode = DoseDisplayMode::Gated4d;
        self.beta = CHALLENGE_IMRT_BETA;
        self.width = IMRT_GRID_DIM;
        self.height = IMRT_GRID_DIM;
        self.prescription_gy = CHALLENGE_IMRT_PRESCRIPTION_GY;
        self.oar_limit_gy = CHALLENGE_IMRT_OAR_LIMIT_GY;
        self.beam_angles_deg = OAR_SPARING_4D_ANGLES_DEG;
        self.beam_weights = OAR_SPARING_4D_WEIGHTS;
        self.imrt_workspace.reset_4d_dose();
        self.refresh_model();
    }

    /// Applies one projected gradient-descent step on the 4 beam weights via `optimization::update_weights`.
    pub fn apply_gradient_weight_hint(&mut self) {
        self.beam_weights = self.imrt_workspace.one_step_weight_hint(
            &self.beam_weights,
            self.prescription_gy,
            self.oar_limit_gy,
        );
        self.imrt_workspace.reset_4d_dose();
        self.refresh_model();
    }

    /// Ingests a `GatingTelemetry` snapshot into the 4D workspace with zero heap allocations.
    pub fn ingest_telemetry_in_place(&mut self, snap: GatingTelemetry) {
        self.latest_telemetry = snap;
        if snap.beam_state == BeamState::SafetyInterlock {
            crate::audio::notify_solver_diverged();
        }
        if self.imrt_mode && self.gating_4d_enabled {
            self.imrt_workspace.step_4d_delivery_in_place(
                &snap,
                self.prescription_gy,
                self.beam_current_ma,
            );
            if snap.step_event && self.texture.is_some() {
                self.texture = None;
            }
        }
    }

    /// Drains all pending `GatingTelemetry` snapshots from `link_bus` with zero heap allocations.
    pub fn poll_link_bus_in_place(&mut self) -> usize {
        let mut step_events = 0_usize;
        let mut batch = [GatingTelemetry::empty(); 16];
        loop {
            let drained = drain_gating_telemetry_into(&mut self.bus_cursor_seq, &mut batch);
            if drained == 0 {
                break;
            }
            self.linked_to_radar_bus = true;
            for snap in batch.iter().copied().take(drained) {
                if snap.step_event {
                    step_events += 1;
                }
                self.ingest_telemetry_in_place(snap);
            }
            if drained < batch.len() {
                break;
            }
        }
        step_events
    }

    /// Advances one 4D respiratory delivery tick from `link_bus` or `Standalone4dStepper`.
    pub fn step_4d_in_place(&mut self) -> GatingTelemetry {
        if self.poll_link_bus_in_place() > 0 {
            return self.latest_telemetry;
        }
        self.linked_to_radar_bus = false;
        let snap = self.standalone_4d.step_zero_alloc();
        self.ingest_telemetry_in_place(snap);
        snap
    }

    /// Resets 4D accumulated dose vectors and standalone respiratory state.
    pub fn reset_4d_simulation(&mut self) {
        self.playing_4d = false;
        self.standalone_4d.reset();
        self.latest_telemetry = self.standalone_4d.latest;
        self.imrt_workspace.reset_4d_dose();
        self.texture = None;
    }

    pub(crate) fn compute_dose_image_for_contrast(&self, high_contrast: bool) -> ColorImage {
        canvas::compute_dose_image_for_contrast(self, high_contrast)
    }
}

impl InteractiveTool for DoseCalculationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
            .or_else(|| self.brush_kit.focused_plot_id())
    }

    fn name(&self) -> &'static str {
        "Dose Calculation"
    }

    fn domain(&self) -> &'static str {
        "medical"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::scalar_field()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        Some(self.compute_dose_image_for_contrast(self.high_contrast))
    }

    fn export_scalar_field(&self) -> Option<ScalarField2D> {
        Some(self.compute_dose_scalar_field(self.width.clamp(60, 256), self.height.clamp(60, 256)))
    }

    fn export_scalar_field_at(&self, width: usize, height: usize) -> Option<ScalarField2D> {
        Some(self.compute_dose_scalar_field(width.clamp(16, 2048), height.clamp(16, 2048)))
    }

    fn state_bounds(&self) -> ToolStateBounds {
        DoseState::STATE_BOUNDS
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if !self.imrt_mode || self.active_challenge.is_some() {
            return;
        }
        if let Some(stroke) = ctx.brush_stroke {
            if self.imrt_workspace.apply_brush_stroke(&stroke) {
                self.refresh_model();
            }
        }
    }

    fn step_simulation(&mut self) {
        if self.imrt_mode && self.gating_4d_enabled {
            let _ = self.step_4d_in_place();
        }
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.amplitude.to_bits().hash(&mut h);
        self.beta.to_bits().hash(&mut h);
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        self.imrt_mode.hash(&mut h);
        self.gating_4d_enabled.hash(&mut h);
        self.beam_current_ma.to_bits().hash(&mut h);
        for b in 0..IMRT_BEAM_COUNT {
            self.beam_angles_deg[b].to_bits().hash(&mut h);
            self.beam_weights[b].to_bits().hash(&mut h);
        }
        self.prescription_gy.to_bits().hash(&mut h);
        self.oar_limit_gy.to_bits().hash(&mut h);
        self.active_challenge.hash(&mut h);
        self.imrt_workspace.custom_contours.hash(&mut h);
        if self.imrt_workspace.custom_contours {
            self.imrt_workspace.tumor_indices.hash(&mut h);
            self.imrt_workspace.organ_indices.hash(&mut h);
        }
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let (ptv_rle, oar_rle) = if self.imrt_mode && self.active_challenge.is_none() {
            self.imrt_workspace.encode_custom_rle_masks()
        } else {
            (None, None)
        };
        let include_current = self.imrt_mode
            && (self.gating_4d_enabled
                || (self.beam_current_ma - NOMINAL_BEAM_CURRENT_MA).abs() > 1.0e-6);
        let state = DoseState {
            amplitude: self.amplitude,
            beta: self.beta,
            width: self.width,
            height: self.height,
            imrt_mode: self.imrt_mode,
            beam_angles_deg: self.imrt_mode.then_some(self.beam_angles_deg),
            beam_weights: self.imrt_mode.then_some(self.beam_weights),
            prescription_gy: self.imrt_mode.then_some(self.prescription_gy),
            oar_limit_gy: self.imrt_mode.then_some(self.oar_limit_gy),
            gating_4d: self.imrt_mode && self.gating_4d_enabled,
            beam_current_ma: include_current.then_some(self.beam_current_ma),
            ptv_rle,
            oar_rle,
        };
        serde_json::to_string(&state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let loaded: DoseState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid DoseCalculation state: {e}"))?;
        loaded.validate()?;
        self.amplitude = loaded.amplitude;
        self.beta = loaded.beta;
        self.width = loaded.width;
        self.height = loaded.height;
        self.imrt_mode = loaded.imrt_mode;
        self.gating_4d_enabled = loaded.imrt_mode && loaded.gating_4d;
        self.display_mode = if self.gating_4d_enabled {
            DoseDisplayMode::Gated4d
        } else {
            DoseDisplayMode::StaticPlan
        };
        self.beam_current_ma = loaded.beam_current_ma.unwrap_or(NOMINAL_BEAM_CURRENT_MA);
        self.beam_angles_deg = loaded.beam_angles_deg.unwrap_or(BASELINE_IMRT_ANGLES_DEG);
        self.beam_weights = loaded.beam_weights.unwrap_or(BASELINE_IMRT_WEIGHTS);
        self.prescription_gy = loaded
            .prescription_gy
            .unwrap_or(CHALLENGE_IMRT_PRESCRIPTION_GY);
        self.oar_limit_gy = loaded.oar_limit_gy.unwrap_or(CHALLENGE_IMRT_OAR_LIMIT_GY);
        self.imrt_workspace
            .restore_rle_masks(loaded.ptv_rle.as_deref(), loaded.oar_rle.as_deref())?;
        self.refresh_model();
        Ok(())
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.refresh_model();
                Ok(())
            }
            Some(CHALLENGE_IMRT_OAR_SPARING) => {
                self.active_challenge = Some(CHALLENGE_IMRT_OAR_SPARING);
                self.imrt_mode = true;
                self.refresh_model();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for DoseCalculationTool"
            )),
        }
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge
            .map(|_| self.imrt_workspace.challenge_evaluation(self.oar_limit_gy))
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        ui::show_tool_inside(self, parent_ui);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "DoseCalculationTool",
        display_name: "Dose Calculation",
        domain: "medical",
        tags: &[
            "psf",
            "convolution",
            "radiation therapy",
            "radiotherapy",
            "isodose",
            "kernel",
            "imrt",
            "gating",
            "contouring",
            "dosimetry",
        ],
        build: || Box::new(DoseCalculationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for DoseCalculationTool {
    fn theory_description(&self) -> String {
        format!(
            "Radiation therapy dose distribution computed via point spread function (PSF) kernel convolution: $D(\\vec{{r}}) = \\int \\Phi(\\vec{{r}}') K(\\vec{{r}} - \\vec{{r}}') d^3\\vec{{r}}'$, using an exponential kernel $K(r) = \\frac{{A}}{{r^2}} e^{{-\\beta r}}$ with amplitude $A = {:.2}$ and attenuation coefficient $\\beta = {:.2}$, coupled with 4D respiratory tissue-shift accumulation and LINAC beam-loading scaling $E(I_b) = E_0 - R_b I_b$.",
            self.amplitude, self.beta
        )
    }
    fn phonetic_description(&self) -> String {
        "Radiation therapy dose calculation, contour sculpting, and four-dimensional respiratory-gated inverse IMRT optimization".into()
    }
    fn theory_citation(&self) -> String {
        "The Physics of Radiation Therapy (Khan, 2014); Convolution/superposition dose calculation (Mackie et al., 1985); AAPM TG-76 Respiratory Motion Management (Keall et al., 2006)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            (
                "amplitude".into(),
                "Kernel primary energy fluence scaling factor A".into(),
            ),
            (
                "beta".into(),
                "Radial dose attenuation and scattering decay coefficient beta".into(),
            ),
        ])
    }
}
