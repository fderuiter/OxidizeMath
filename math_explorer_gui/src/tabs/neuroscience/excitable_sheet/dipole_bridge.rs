#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{LeadCoord, ScenarioPreset};

#[cfg(feature = "physics")]
use math_explorer::physics::medical::bioelectric::{
    BioelectricPresetKind, DipoleLeadArray, LeadPosition, LeadReadoutMode,
};

#[cfg(not(feature = "physics"))]
use math_explorer::math_kernel::types::flatten_2d_index;

/// Feature-gated bridge wrapping `domain_physics::physics::medical::bioelectric` when `physics`
/// is enabled and providing an isolated-build fallback when `physics` is disabled.
pub struct LeadIntegratorBridge {
    #[cfg(feature = "physics")]
    inner: DipoleLeadArray,
    #[cfg(not(feature = "physics"))]
    width: usize,
    #[cfg(not(feature = "physics"))]
    height: usize,
    #[cfg(not(feature = "physics"))]
    leads: [LeadCoord; 3],
}

impl LeadIntegratorBridge {
    /// Creates a pre-allocated 3-lead dipole integrator for a `width x height` sheet.
    pub fn new(width: usize, height: usize, leads: [LeadCoord; 3]) -> Self {
        #[cfg(feature = "physics")]
        {
            let mut inner = DipoleLeadArray::new_default(width, height);
            for (i, c) in leads.into_iter().enumerate() {
                inner.set_lead(i, LeadPosition::new(c.x, c.y, c.z));
            }
            Self { inner }
        }
        #[cfg(not(feature = "physics"))]
        {
            Self {
                width,
                height,
                leads,
            }
        }
    }

    /// Updates a single virtual lead coordinate in place.
    pub fn set_lead(&mut self, index: usize, coord: LeadCoord) {
        if index >= 3 {
            return;
        }
        #[cfg(feature = "physics")]
        {
            self.inner
                .set_lead(index, LeadPosition::new(coord.x, coord.y, coord.z));
        }
        #[cfg(not(feature = "physics"))]
        {
            self.leads[index] = coord;
        }
    }

    /// Updates all three virtual leads in place.
    pub fn sync_leads(&mut self, leads: [LeadCoord; 3]) {
        for (i, c) in leads.into_iter().enumerate() {
            self.set_lead(i, c);
        }
    }

    /// Returns canonical shared lead coordinates for `preset` on a `width x height` sheet.
    pub fn shared_preset_leads(
        preset: ScenarioPreset,
        width: usize,
        height: usize,
    ) -> ([LeadCoord; 3], bool) {
        #[cfg(feature = "physics")]
        {
            let kind = match preset {
                ScenarioPreset::DualEdgeCollision => BioelectricPresetKind::DualEdgeCollision,
                ScenarioPreset::S1S2SpiralInduction => BioelectricPresetKind::S1S2SpiralInduction,
                ScenarioPreset::ScarPinnedRotor => BioelectricPresetKind::ScarPinningAblation,
                ScenarioPreset::WholeSheetDefibrillation => {
                    BioelectricPresetKind::WholeSheetDefibrillation
                }
            };
            let spec = kind.spec(width, height);
            let coords = spec.leads.map(|l| LeadCoord::new(l.x, l.y, l.z));
            let bipolar = spec.readout_mode == LeadReadoutMode::BipolarPseudoEcg;
            (coords, bipolar)
        }
        #[cfg(not(feature = "physics"))]
        {
            let w = width as f64;
            let h = height as f64;
            match preset {
                ScenarioPreset::DualEdgeCollision => (
                    [
                        LeadCoord::new(w * 0.22, h * 0.50, 8.0),
                        LeadCoord::new(w * 0.50, h * 0.50, 8.0),
                        LeadCoord::new(w * 0.78, h * 0.50, 8.0),
                    ],
                    false,
                ),
                ScenarioPreset::ScarPinnedRotor => (
                    [
                        LeadCoord::new(w * 0.24, h * 0.26, 7.5),
                        LeadCoord::new(w * 0.76, h * 0.26, 7.5),
                        LeadCoord::new(w * 0.50, h * 0.82, 7.5),
                    ],
                    true,
                ),
                ScenarioPreset::S1S2SpiralInduction | ScenarioPreset::WholeSheetDefibrillation => (
                    [
                        LeadCoord::new(w * 0.18, h * 0.20, 8.0),
                        LeadCoord::new(w * 0.82, h * 0.20, 8.0),
                        LeadCoord::new(w * 0.50, h * 0.86, 8.0),
                    ],
                    true,
                ),
            }
        }
    }

    /// Evaluates 3-lead potentials across a 2D transmembrane voltage sheet.
    pub fn evaluate_sheet(&self, voltage: &[f64], scar_mask: &[u8], bipolar: bool) -> [f64; 3] {
        #[cfg(feature = "physics")]
        {
            let mode = if bipolar {
                LeadReadoutMode::BipolarPseudoEcg
            } else {
                LeadReadoutMode::UnipolarLfp
            };
            self.inner.evaluate_sheet(voltage, scar_mask, mode)
        }
        #[cfg(not(feature = "physics"))]
        {
            self.evaluate_sheet_fallback(voltage, scar_mask, bipolar)
        }
    }

    /// Evaluates 3-lead potentials along a 1D diffusive cable.
    pub fn evaluate_cable(&self, voltage: &[f64], scar_mask: &[u8], bipolar: bool) -> [f64; 3] {
        #[cfg(feature = "physics")]
        {
            let mode = if bipolar {
                LeadReadoutMode::BipolarPseudoEcg
            } else {
                LeadReadoutMode::UnipolarLfp
            };
            self.inner
                .evaluate_cable_1d(voltage, scar_mask, 48.0, 0.5, mode)
        }
        #[cfg(not(feature = "physics"))]
        {
            let mut phi = [0.0; 3];
            for i in 1..voltage.len().saturating_sub(1) {
                if scar_mask[i] != 0 {
                    continue;
                }
                let dv = voltage[i + 1] - voltage[i - 1];
                let px = i as f64;
                for (k, lead) in self.leads.iter().enumerate() {
                    let rx = px - lead.x;
                    let ry = 48.0 - lead.y;
                    let r2 = rx * rx + ry * ry + lead.z * lead.z;
                    phi[k] += dv * rx / (r2 * r2.sqrt());
                }
            }
            if bipolar {
                [phi[1] - phi[0], phi[2] - phi[0], phi[2] - phi[1]]
            } else {
                phi
            }
        }
    }

    #[cfg(not(feature = "physics"))]
    fn evaluate_sheet_fallback(
        &self,
        voltage: &[f64],
        scar_mask: &[u8],
        bipolar: bool,
    ) -> [f64; 3] {
        let mut phi = [0.0; 3];
        for y in 1..self.height.saturating_sub(1) {
            for x in 1..self.width.saturating_sub(1) {
                let idx = flatten_2d_index(x, y, self.width);
                if scar_mask[idx] != 0 {
                    continue;
                }
                let l = flatten_2d_index(x - 1, y, self.width);
                let r = flatten_2d_index(x + 1, y, self.width);
                let u = flatten_2d_index(x, y - 1, self.width);
                let d = flatten_2d_index(x, y + 1, self.width);
                let gx = 0.5 * (voltage[r] - voltage[l]);
                let gy = 0.5 * (voltage[d] - voltage[u]);
                for (k, lead) in self.leads.iter().enumerate() {
                    let rx = x as f64 - lead.x;
                    let ry = y as f64 - lead.y;
                    let r2 = rx * rx + ry * ry + lead.z * lead.z;
                    let inv_r3 = 25.0 / (r2 * r2.sqrt());
                    phi[k] += (gx * rx + gy * ry) * inv_r3;
                }
            }
        }
        if bipolar {
            [phi[1] - phi[0], phi[2] - phi[0], phi[2] - phi[1]]
        } else {
            phi
        }
    }
}
