#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::biology::neuroscience::DEFAULT_SHEET_DIM;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Spatial tissue propagation engine mode inside `ExcitableSheetTool`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationMode {
    /// 2D `96x96` FitzHugh-Nagumo reaction-diffusion sheet (`FiniteDifference2D` + `FusedEulerSolver`).
    FitzHughNagumo2D,
    /// 1D `96`-compartment diffusive Hodgkin-Huxley squid-axon cable (`FiniteDifference1D`).
    HodgkinHuxleyCable1D,
}

/// Tactile pointer and keyboard brush operation on the excitable canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrushMode {
    /// Injects depolarizing or hyperpolarizing current ($I_{\text{stim}}$).
    InjectCurrent,
    /// Holds local membrane potential fixed at `clamp_voltage`.
    ClampVoltage,
    /// Paints zero-flux insulating ischemic scar or catheter ablation tissue.
    PaintScar,
    /// Erases insulating scar tissue back to excitable myocardium/cortex.
    EraseScar,
    /// Drags the active virtual extracellular sensing lead across the sheet.
    MoveLead,
}

/// Voltage heatmap palette selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColormapMode {
    /// Perceptual thermal ramp (`heatmap_color`).
    Perceptual,
    /// WCAG high-contrast dark-red -> black -> cyan ramp (`heatmap_color_high_contrast`).
    HighContrast,
}

/// Starter electrophysiology scenario configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioPreset {
    /// Dual-edge planar wavefronts demonstrating head-on collision annihilation.
    DualEdgeCollision,
    /// Timed S1-S2 stimulation inducing a self-sustaining spiral arrhythmia.
    S1S2SpiralInduction,
    /// Re-entrant rotor pinned around a central zero-flux ischemic scar.
    ScarPinnedRotor,
    /// Synchronous whole-sheet defibrillation shock restoring resting equilibrium.
    WholeSheetDefibrillation,
}

/// Serializable 3D coordinate for a virtual extracellular lead.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeadCoord {
    /// Horizontal sheet coordinate (`0.0..=96.0`).
    pub x: f64,
    /// Vertical sheet coordinate (`0.0..=96.0`).
    pub y: f64,
    /// Stand-off elevation above the tissue plane (`0.5..=64.0`).
    pub z: f64,
}

impl LeadCoord {
    /// Creates a validated virtual lead coordinate.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Complete validated state schema for `ExcitableSheetTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcitableSheetState {
    /// FitzHugh-Nagumo recovery offset `a` (`0.1..=1.5`).
    pub a: f64,
    /// FitzHugh-Nagumo recovery linear damping `b` (`0.1..=1.5`).
    pub b: f64,
    /// FitzHugh-Nagumo slow recovery timescale `epsilon` (`0.005..=0.5`).
    pub epsilon: f64,
    /// Homogeneous background transmembrane current `i_ext` (`-1.0..=2.0`).
    pub i_ext: f64,
    /// Voltage diffusion coefficient `d_u` (`0.05..=2.0`).
    pub d_u: f64,
    /// Integration time step `dt` (`0.005..=0.25`).
    pub dt: f64,
    /// Sub-steps executed per UI frame (`1..=16`).
    pub steps_per_frame: usize,
    /// Brush disc radius in grid cells (`0.5..=16.0`).
    pub brush_radius: f64,
    /// Injected electrode current amplitude (`-3.0..=5.0`).
    pub brush_current: f64,
    /// Voltage-clamp target potential (`-80.0..=40.0`).
    pub clamp_voltage: f64,
    /// Active propagation model (2D FHN sheet vs 1D HH cable).
    pub propagation_mode: PropagationMode,
    /// Active pointer/keyboard brush mode.
    pub brush_mode: BrushMode,
    /// Active heatmap colormap mode.
    pub colormap: ColormapMode,
    /// `true` for 3-lead bipolar Einthoven pseudo-ECG, `false` for unipolar LFP.
    pub readout_bipolar: bool,
    /// Selected virtual lead index (`0..=2`).
    pub active_lead: usize,
    /// Spatial coordinates of the three virtual extracellular leads.
    pub leads: [LeadCoord; 3],
    /// Active starter scenario preset.
    pub scenario: ScenarioPreset,
    /// Whether the catheter ablation line to the right sheet boundary is active.
    pub ablated_to_edge: bool,
    /// Whether frame-by-frame time integration is actively running.
    pub running: bool,
    /// Optional `base64url` RLE (`RM` v1) spatial mask persisting custom painted/erased scars.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scar_mask_rle: Option<String>,
}

impl Default for ExcitableSheetState {
    fn default() -> Self {
        Self {
            a: 0.7,
            b: 0.8,
            epsilon: 0.06,
            i_ext: 0.0,
            d_u: 0.5,
            dt: 0.15,
            steps_per_frame: 4,
            brush_radius: 3.5,
            brush_current: 2.2,
            clamp_voltage: 1.6,
            propagation_mode: PropagationMode::FitzHughNagumo2D,
            brush_mode: BrushMode::InjectCurrent,
            colormap: ColormapMode::Perceptual,
            readout_bipolar: true,
            active_lead: 0,
            leads: [
                LeadCoord::new(17.28, 19.20, 8.0),
                LeadCoord::new(78.72, 19.20, 8.0),
                LeadCoord::new(48.00, 82.56, 8.0),
            ],
            scenario: ScenarioPreset::S1S2SpiralInduction,
            ablated_to_edge: false,
            running: false,
            scar_mask_rle: None,
        }
    }
}

impl ExcitableSheetState {
    /// Validates finiteness, domain bounds, and pre-allocation RLE scar mask integrity.
    pub fn validate(&self) -> Result<(), String> {
        let scalar_bounds = [
            ("a", self.a, 0.1, 1.5),
            ("b", self.b, 0.1, 1.5),
            ("epsilon", self.epsilon, 0.005, 0.5),
            ("i_ext", self.i_ext, -1.0, 2.0),
            ("d_u", self.d_u, 0.05, 2.0),
            ("dt", self.dt, 0.005, 0.25),
            ("brush_radius", self.brush_radius, 0.5, 16.0),
            ("brush_current", self.brush_current, -3.0, 5.0),
            ("clamp_voltage", self.clamp_voltage, -80.0, 40.0),
        ];
        for (name, val, lo, hi) in scalar_bounds {
            if !val.is_finite() || val < lo || val > hi {
                return Err(format!(
                    "ExcitableSheet parameter '{name}' = {val} must be finite within [{lo}, {hi}]"
                ));
            }
        }
        if self.steps_per_frame == 0 || self.steps_per_frame > 16 {
            return Err("steps_per_frame must be within 1..=16".to_string());
        }
        if self.active_lead >= 3 {
            return Err("active_lead must be 0, 1, or 2".to_string());
        }
        for (idx, lead) in self.leads.iter().enumerate() {
            if !lead.x.is_finite()
                || !lead.y.is_finite()
                || !lead.z.is_finite()
                || !(0.0..=96.0).contains(&lead.x)
                || !(0.0..=96.0).contains(&lead.y)
                || !(0.5..=64.0).contains(&lead.z)
            {
                return Err(format!("Lead {idx} coordinates out of bounds"));
            }
        }
        if let Some(ref rle) = self.scar_mask_rle {
            let _ = crate::framework::raster_brush::decode_u8_grid(
                rle,
                DEFAULT_SHEET_DIM,
                DEFAULT_SHEET_DIM,
                1,
            )?;
        }
        Ok(())
    }

    /// Clamps any out-of-range scalar parameters or invalid scar RLE masks back into stable bounds.
    pub fn clamp_in_place(&mut self) {
        fn clamp_or(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
            if val.is_finite() {
                val.clamp(lo, hi)
            } else {
                fallback
            }
        }
        self.a = clamp_or(self.a, 0.1, 1.5, 0.7);
        self.b = clamp_or(self.b, 0.1, 1.5, 0.8);
        self.epsilon = clamp_or(self.epsilon, 0.01, 0.2, 0.06);
        self.i_ext = clamp_or(self.i_ext, -1.0, 2.0, 0.0);
        self.d_u = clamp_or(self.d_u, 0.05, 1.2, 0.5);
        self.dt = clamp_or(self.dt, 0.01, 0.2, 0.15);
        self.steps_per_frame = self.steps_per_frame.clamp(1, 16);
        self.brush_radius = clamp_or(self.brush_radius, 0.5, 16.0, 3.5);
        self.brush_current = clamp_or(self.brush_current, -3.0, 5.0, 2.2);
        self.clamp_voltage = clamp_or(self.clamp_voltage, -80.0, 40.0, 1.6);
        self.active_lead = self.active_lead.min(2);
        for lead in &mut self.leads {
            lead.x = clamp_or(lead.x, 0.0, 96.0, 48.0);
            lead.y = clamp_or(lead.y, 0.0, 96.0, 48.0);
            lead.z = clamp_or(lead.z, 0.5, 64.0, 8.0);
        }
        if let Some(ref rle) = self.scar_mask_rle {
            if crate::framework::raster_brush::decode_u8_grid(
                rle,
                DEFAULT_SHEET_DIM,
                DEFAULT_SHEET_DIM,
                1,
            )
            .is_err()
            {
                self.scar_mask_rle = None;
            }
        }
    }
}

/// Theory portal metadata provider for `ExcitableSheetTool`.
#[derive(Default)]
pub struct ExcitableSheetTheory;

impl scientific_metadata::theory::TheoryDescribable for ExcitableSheetTheory {
    fn theory_description(&self) -> String {
        "Couples 2D FitzHugh-Nagumo reaction-diffusion electrophysiology (∂u/∂t = D_u∇²u + u - u³/3 - w + I_ext, ∂w/∂t = ε(u + a - bw)) and a 1D diffusive Hodgkin-Huxley cable with three movable extracellular dipole leads Φ_k ∝ ∑(∇V · d_k)/‖d_k‖³.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Partial u partial t equals D u Laplacian u plus u minus u cubed over three minus w plus external current; extracellular potential Phi k equals the sum of gradient V dot d k divided by norm d k cubed.".to_string()
    }

    fn theory_citation(&self) -> String {
        "FitzHugh (1961), Biophysical Journal 1(6), 445-466; Nagumo et al. (1962), Proc. IRE 50(10), 2061-2070; Hodgkin & Huxley (1952), J. Physiol. 117(4), 500-544.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn maturity_tier(&self) -> scientific_metadata::theory::MaturityTier {
        scientific_metadata::theory::MaturityTier::Certified
    }
}
