//! Shared clinical electrophysiology presets and standalone wave-field synthesizers for 3-lead mapping.

use super::bioelectric::{DipoleLeadArray, LeadPosition, LeadReadoutMode};
use math_commons::math_kernel::types::flatten_2d_index;
use serde::{Deserialize, Serialize};

/// Shared cross-domain clinical electrophysiology preset identifier.
// [cite:medical]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BioelectricPresetKind {
    /// Opposing dual-edge planar wavefronts demonstrating refractory collision annihilation.
    DualEdgeCollision,
    /// S1-S2 vulnerability-window stimulation spawning a re-entrant spiral tachycardia.
    S1S2SpiralInduction,
    /// Ischemic zero-flux scar anchoring a rotor until catheter ablation reaches the boundary.
    ScarPinningAblation,
    /// Synchronous whole-sheet defibrillation shock restoring isoelectric resting equilibrium.
    WholeSheetDefibrillation,
}

/// Shared configuration describing lead placement, readout mode, and scar geometry for a preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BioelectricPresetSpec {
    /// Preset variant.
    pub kind: BioelectricPresetKind,
    /// Display title shared across `NeuroscienceTab` and `MedicalTab`.
    pub title: &'static str,
    /// Three virtual electrode coordinates on a `96x96` reference sheet.
    pub leads: [LeadPosition; 3],
    /// Recommended extracellular readout derivation mode.
    pub readout_mode: LeadReadoutMode,
    /// Optional central ischemic scar radius (`0.0` when unscarred).
    pub central_scar_radius: f64,
    /// Whether an ablation line connects the central scar to the right boundary.
    pub ablated_to_edge: bool,
}

impl BioelectricPresetKind {
    /// Returns the shared preset specification on a `width x height` sheet.
    #[verified_engine::verified]
    pub fn spec(self, width: usize, height: usize) -> BioelectricPresetSpec {
        let w = width.max(4) as f64;
        let h = height.max(4) as f64;
        match self {
            Self::DualEdgeCollision => BioelectricPresetSpec {
                kind: self,
                title: "Dual-Edge Wavefront Collision",
                leads: [
                    LeadPosition::new(w * 0.22, h * 0.50, 8.0),
                    LeadPosition::new(w * 0.50, h * 0.50, 8.0),
                    LeadPosition::new(w * 0.78, h * 0.50, 8.0),
                ],
                readout_mode: LeadReadoutMode::UnipolarLfp,
                central_scar_radius: 0.0,
                ablated_to_edge: false,
            },
            Self::S1S2SpiralInduction => BioelectricPresetSpec {
                kind: self,
                title: "S1-S2 Spiral Arrhythmia",
                leads: DipoleLeadArray::default_einthoven_leads(width, height),
                readout_mode: LeadReadoutMode::BipolarPseudoEcg,
                central_scar_radius: 0.0,
                ablated_to_edge: false,
            },
            Self::ScarPinningAblation => BioelectricPresetSpec {
                kind: self,
                title: "Scar-Pinned Rotor & Ablation",
                leads: [
                    LeadPosition::new(w * 0.24, h * 0.26, 7.5),
                    LeadPosition::new(w * 0.76, h * 0.26, 7.5),
                    LeadPosition::new(w * 0.50, h * 0.82, 7.5),
                ],
                readout_mode: LeadReadoutMode::BipolarPseudoEcg,
                central_scar_radius: (w.min(h) * 0.14).max(4.0),
                ablated_to_edge: false,
            },
            Self::WholeSheetDefibrillation => BioelectricPresetSpec {
                kind: self,
                title: "Whole-Sheet Defibrillation",
                leads: DipoleLeadArray::default_einthoven_leads(width, height),
                readout_mode: LeadReadoutMode::BipolarPseudoEcg,
                central_scar_radius: 0.0,
                ablated_to_edge: false,
            },
        }
    }
}

/// Populates `voltage` in place with a smooth kinematic transmembrane wave for standalone `MedicalTab` lead mapping.
#[verified_engine::verified]
pub fn synthesize_bioelectric_preset_sheet(
    voltage: &mut [f64],
    scar_mask: &[u8],
    width: usize,
    height: usize,
    preset: BioelectricPresetKind,
    phase: f64,
) {
    let n = width * height;
    if voltage.len() < n || scar_mask.len() < n {
        return;
    }
    let cx = width as f64 * 0.5;
    let cy = height as f64 * 0.5;
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            if scar_mask[idx] != 0 {
                voltage[idx] = -1.2;
                continue;
            }
            voltage[idx] = sample_kinematic_voltage(preset, x as f64, y as f64, cx, cy, phase);
        }
    }
}

#[verified_engine::verified]
fn sample_kinematic_voltage(
    preset: BioelectricPresetKind,
    x: f64,
    y: f64,
    cx: f64,
    cy: f64,
    phase: f64,
) -> f64 {
    match preset {
        BioelectricPresetKind::WholeSheetDefibrillation => {
            let decay = (-phase.max(0.0) * 1.4).exp();
            -1.2 + 2.6 * decay
        }
        BioelectricPresetKind::DualEdgeCollision => {
            let travel = (phase.rem_euclid(6.0)) * (cx * 0.24);
            if travel > cx + 4.0 {
                return -1.2;
            }
            let d_left = (x - travel).abs();
            let d_right = ((2.0 * cx - travel) - x).abs();
            let pulse = (-0.08 * d_left * d_left)
                .exp()
                .max((-0.08 * d_right * d_right).exp());
            -1.2 + 3.0 * pulse
        }
        BioelectricPresetKind::S1S2SpiralInduction => {
            let dx = x - cx;
            let dy = y - cy;
            let r = dx.hypot(dy);
            let theta = dy.atan2(dx);
            let wave = (theta + 0.22 * r - phase * 2.2).cos();
            let core = (r / 6.0).tanh();
            -1.2 + 1.5 * (1.0 + wave * core)
        }
        BioelectricPresetKind::ScarPinningAblation => {
            let dx = x - cx;
            let dy = y - cy;
            let theta = dy.atan2(dx);
            let wave = (theta - phase * 2.0).cos();
            -1.2 + 1.5 * (1.0 + wave)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_dipole_lead_polarity_inverts_across_wavefront_and_obeys_einthoven() {
        let w = 96;
        let h = 96;
        let scars = vec![0_u8; w * h];
        let mut voltage = vec![-1.2; w * h];
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                voltage[idx] = if x < 48 { 1.8 } else { -1.2 };
            }
        }
        let mut leads = DipoleLeadArray::new_default(w, h);
        leads.set_lead(0, LeadPosition::new(68.0, 48.0, 8.0));
        leads.set_lead(1, LeadPosition::new(28.0, 48.0, 8.0));
        let lfp = leads.evaluate_sheet(&voltage, &scars, LeadReadoutMode::UnipolarLfp);
        assert!(
            lfp[0] > 0.0,
            "Lead ahead of depolarizing front must be positive: {}",
            lfp[0]
        );
        assert!(
            lfp[1] < 0.0,
            "Lead behind depolarizing front must be negative: {}",
            lfp[1]
        );

        let ecg = DipoleLeadArray::to_bipolar_ecg(lfp);
        assert!((ecg[1] - (ecg[0] + ecg[2])).abs() < 1e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_cable_1d_dipole_polarity_inversion() {
        let mut v_1d = vec![-65.0; 96];
        for v in v_1d.iter_mut().take(48) {
            *v = 25.0;
        }
        let scars_1d = vec![0_u8; 96];
        let mut leads = DipoleLeadArray::new_default(96, 96);
        leads.set_lead(0, LeadPosition::new(72.0, 48.0, 6.0));
        leads.set_lead(1, LeadPosition::new(24.0, 48.0, 6.0));
        let lfp =
            leads.evaluate_cable_1d(&v_1d, &scars_1d, 48.0, 0.5, LeadReadoutMode::UnipolarLfp);
        assert!(lfp[0] > 0.0 && lfp[1] < 0.0);
    }
}
