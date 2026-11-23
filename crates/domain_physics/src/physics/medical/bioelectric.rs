//! Extracellular 3-lead bioelectric dipole potential integration for excitable tissue sheets.
//!
//! Evaluates quasi-static volume-conductor dipole potentials across three movable virtual leads:
//! $$ \Phi_k = \kappa \sum_{x, y} \frac{\nabla V(x, y) \cdot \mathbf{d}_k(x, y)}{\|\mathbf{d}_k(x, y)\|^3} \, \Delta x \Delta y $$
//! where $\mathbf{d}_k(x, y) = (x - x_k, \, y - y_k, \, -z_k)$, yielding positive upward R-wave
//! deflections as depolarizing wavefronts approach lead $k$ and negative deflections as they recede.

use math_commons::math_kernel::types::flatten_2d_index;
use serde::{Deserialize, Serialize};

pub use super::bioelectric_presets::{
    BioelectricPresetKind, BioelectricPresetSpec, synthesize_bioelectric_preset_sheet,
};

/// 3D spatial coordinates of an extracellular virtual sensing lead relative to the tissue sheet.
// [cite:medical]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeadPosition {
    /// Horizontal position in grid units (`[0, width]`).
    pub x: f64,
    /// Vertical position in grid units (`[0, height]`).
    pub y: f64,
    /// Perpendicular stand-off height above the tissue plane (`z > 0`).
    pub z: f64,
}

impl LeadPosition {
    /// Constructs a virtual lead position with stand-off height clamped above `0.5` grid units.
    #[verified_engine::verified]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z: z.max(0.5),
        }
    }
}

/// Extracellular readout derivation mode across the three virtual electrodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeadReadoutMode {
    /// Bipolar Einthoven pseudo-ECG leads `[Lead I = Φ2 - Φ1, Lead II = Φ3 - Φ1, Lead III = Φ3 - Φ2]`.
    BipolarPseudoEcg,
    /// Unipolar extracellular local field potentials `[Φ1, Φ2, Φ3]`.
    UnipolarLfp,
}

/// Pre-allocated 3-lead dipole integrator for 2D sheets and 1D cables.
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::medical::bioelectric::{DipoleLeadArray, LeadReadoutMode};
///
/// let leads = DipoleLeadArray::new_default(96, 96);
/// let voltage = vec![-1.2; 96 * 96];
/// let scars = vec![0_u8; 96 * 96];
/// let ecg = leads.evaluate_sheet(&voltage, &scars, LeadReadoutMode::BipolarPseudoEcg);
/// assert!(ecg[0].abs() < 1e-12);
/// ```
// [cite:medical]
#[derive(Debug, Clone)]
pub struct DipoleLeadArray {
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    gain: f64,
    leads: [LeadPosition; 3],
    wx0: Vec<f64>,
    wy0: Vec<f64>,
    wx1: Vec<f64>,
    wy1: Vec<f64>,
    wx2: Vec<f64>,
    wy2: Vec<f64>,
}

impl DipoleLeadArray {
    /// Creates a pre-allocated 3-lead Einthoven triangle array over a `width x height` sheet.
    #[verified_engine::verified]
    pub fn new_default(width: usize, height: usize) -> Self {
        let w = width.max(4);
        let h = height.max(4);
        let default_leads = Self::default_einthoven_leads(w, h);
        Self::new(w, h, 1.0, 1.0, 25.0, default_leads)
    }

    /// Canonical Einthoven-style virtual electrode triangle (`RA`, `LA`, `LL`) over `width x height`.
    #[verified_engine::verified]
    pub fn default_einthoven_leads(width: usize, height: usize) -> [LeadPosition; 3] {
        let w = width.max(4) as f64;
        let h = height.max(4) as f64;
        [
            LeadPosition::new(w * 0.18, h * 0.20, 8.0),
            LeadPosition::new(w * 0.82, h * 0.20, 8.0),
            LeadPosition::new(w * 0.50, h * 0.86, 8.0),
        ]
    }

    /// Constructs a pre-allocated 3-lead dipole integrator and populates all kernel weights.
    #[verified_engine::verified]
    pub fn new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        gain: f64,
        leads: [LeadPosition; 3],
    ) -> Self {
        let w = width.max(4);
        let h = height.max(4);
        let n = w * h;
        let mut array = Self {
            width: w,
            height: h,
            dx: dx.max(1e-3),
            dy: dy.max(1e-3),
            gain,
            leads,
            wx0: vec![0.0; n],
            wy0: vec![0.0; n],
            wx1: vec![0.0; n],
            wy1: vec![0.0; n],
            wx2: vec![0.0; n],
            wy2: vec![0.0; n],
        };
        array.rebuild_all_kernels();
        array
    }

    /// Grid width in cells.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height in cells.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Current 3-lead spatial coordinates.
    #[verified_engine::verified]
    pub fn leads(&self) -> &[LeadPosition; 3] {
        &self.leads
    }

    /// Moves lead `index` (`0..3`) and updates its pre-allocated dipole kernel weights in place.
    #[verified_engine::verified]
    pub fn set_lead(&mut self, index: usize, position: LeadPosition) {
        if index >= 3 {
            return;
        }
        let clamped = LeadPosition::new(
            position.x.clamp(0.0, self.width as f64),
            position.y.clamp(0.0, self.height as f64),
            position.z.clamp(0.5, 64.0),
        );
        if self.leads[index] == clamped {
            return;
        }
        self.leads[index] = clamped;
        self.rebuild_lead_kernel(index);
    }

    /// Updates all three virtual leads and refreshes modified dipole kernels in place.
    #[verified_engine::verified]
    pub fn set_leads(&mut self, leads: [LeadPosition; 3]) {
        for (i, lead) in leads.into_iter().enumerate() {
            self.set_lead(i, lead);
        }
    }

    /// Converts unipolar potentials `[Φ1, Φ2, Φ3]` into bipolar Einthoven `[Lead I, Lead II, Lead III]`.
    #[verified_engine::verified]
    pub fn to_bipolar_ecg(unipolar: [f64; 3]) -> [f64; 3] {
        [
            unipolar[1] - unipolar[0],
            unipolar[2] - unipolar[0],
            unipolar[2] - unipolar[1],
        ]
    }

    /// Integrates 3-lead extracellular potentials across a 2D transmembrane voltage sheet with zero allocations.
    #[verified_engine::verified]
    pub fn evaluate_sheet(
        &self,
        voltage: &[f64],
        scar_mask: &[u8],
        mode: LeadReadoutMode,
    ) -> [f64; 3] {
        let unipolar = self.integrate_unipolar_sheet(voltage, scar_mask);
        match mode {
            LeadReadoutMode::BipolarPseudoEcg => Self::to_bipolar_ecg(unipolar),
            LeadReadoutMode::UnipolarLfp => unipolar,
        }
    }

    /// Integrates unipolar potentials `[Φ1, Φ2, Φ3]` across a 2D voltage field with zero allocations.
    #[verified_engine::verified]
    pub fn integrate_unipolar_sheet(&self, voltage: &[f64], scar_mask: &[u8]) -> [f64; 3] {
        let n = self.width * self.height;
        if voltage.len() < n || scar_mask.len() < n {
            return [0.0; 3];
        }
        let inv_dx = 1.0 / self.dx;
        let inv_dy = 1.0 / self.dy;
        let mut phi = [0.0; 3];

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = flatten_2d_index(x, y, self.width);
                if scar_mask[idx] != 0 {
                    continue;
                }
                let (dv_dx, dv_dy) =
                    self.cell_voltage_gradient(voltage, scar_mask, (x, y), idx, (inv_dx, inv_dy));
                phi[0] += dv_dx * self.wx0[idx] + dv_dy * self.wy0[idx];
                phi[1] += dv_dx * self.wx1[idx] + dv_dy * self.wy1[idx];
                phi[2] += dv_dx * self.wx2[idx] + dv_dy * self.wy2[idx];
            }
        }
        phi
    }

    /// Integrates 3-lead extracellular dipole potentials along a 1D diffusive cable at row `cable_y`.
    #[verified_engine::verified]
    pub fn evaluate_cable_1d(
        &self,
        voltage_1d: &[f64],
        scar_mask_1d: &[u8],
        cable_y: f64,
        dx_1d: f64,
        mode: LeadReadoutMode,
    ) -> [f64; 3] {
        let len = voltage_1d.len().min(scar_mask_1d.len());
        if len < 2 {
            return [0.0; 3];
        }
        let safe_dx = dx_1d.max(1e-3);
        let scale_x = (self.width as f64) / (len as f64);
        let mut unipolar = [0.0; 3];
        for i in 0..len {
            if scar_mask_1d[i] != 0 {
                continue;
            }
            let v_l = if i > 0 && scar_mask_1d[i - 1] == 0 {
                voltage_1d[i - 1]
            } else {
                voltage_1d[i]
            };
            let v_r = if i + 1 < len && scar_mask_1d[i + 1] == 0 {
                voltage_1d[i + 1]
            } else {
                voltage_1d[i]
            };
            let span =
                if i > 0 && i + 1 < len && scar_mask_1d[i - 1] == 0 && scar_mask_1d[i + 1] == 0 {
                    2.0 * safe_dx
                } else {
                    safe_dx
                };
            let dv_dx = (v_r - v_l) / span;
            let pos_x = (i as f64 + 0.5) * scale_x;
            for (k, lead) in self.leads.iter().enumerate() {
                let rx = pos_x - lead.x;
                let ry = cable_y - lead.y;
                let r2 = rx * rx + ry * ry + lead.z * lead.z;
                let inv_r3 = 1.0 / (r2 * r2.sqrt());
                unipolar[k] += 0.08 * self.gain * safe_dx * dv_dx * rx * inv_r3;
            }
        }
        match mode {
            LeadReadoutMode::BipolarPseudoEcg => Self::to_bipolar_ecg(unipolar),
            LeadReadoutMode::UnipolarLfp => unipolar,
        }
    }

    #[verified_engine::verified]
    fn cell_voltage_gradient(
        &self,
        voltage: &[f64],
        scar_mask: &[u8],
        (x, y): (usize, usize),
        idx: usize,
        (inv_dx, inv_dy): (f64, f64),
    ) -> (f64, f64) {
        let v_c = voltage[idx];
        let (v_l, has_l) = if x > 0 {
            let l = flatten_2d_index(x - 1, y, self.width);
            if scar_mask[l] == 0 {
                (voltage[l], true)
            } else {
                (v_c, false)
            }
        } else {
            (v_c, false)
        };
        let (v_r, has_r) = if x + 1 < self.width {
            let r = flatten_2d_index(x + 1, y, self.width);
            if scar_mask[r] == 0 {
                (voltage[r], true)
            } else {
                (v_c, false)
            }
        } else {
            (v_c, false)
        };
        let (v_u, has_u) = if y > 0 {
            let u = flatten_2d_index(x, y - 1, self.width);
            if scar_mask[u] == 0 {
                (voltage[u], true)
            } else {
                (v_c, false)
            }
        } else {
            (v_c, false)
        };
        let (v_d, has_d) = if y + 1 < self.height {
            let d = flatten_2d_index(x, y + 1, self.width);
            if scar_mask[d] == 0 {
                (voltage[d], true)
            } else {
                (v_c, false)
            }
        } else {
            (v_c, false)
        };
        let dv_dx = (v_r - v_l) * inv_dx * if has_l && has_r { 0.5 } else { 1.0 };
        let dv_dy = (v_d - v_u) * inv_dy * if has_u && has_d { 0.5 } else { 1.0 };
        (dv_dx, dv_dy)
    }

    #[verified_engine::verified]
    fn rebuild_all_kernels(&mut self) {
        self.rebuild_lead_kernel(0);
        self.rebuild_lead_kernel(1);
        self.rebuild_lead_kernel(2);
    }

    #[verified_engine::verified]
    fn rebuild_lead_kernel(&mut self, index: usize) {
        let lead = self.leads[index];
        let area_gain = self.gain * self.dx * self.dy;
        let w = self.width;
        let h = self.height;
        let dx = self.dx;
        let dy = self.dy;
        let (wx, wy) = match index {
            0 => (&mut self.wx0, &mut self.wy0),
            1 => (&mut self.wx1, &mut self.wy1),
            _ => (&mut self.wx2, &mut self.wy2),
        };
        for y in 0..h {
            let py = (y as f64 + 0.5) * dy;
            let ry = py - lead.y * dy;
            for x in 0..w {
                let px = (x as f64 + 0.5) * dx;
                let rx = px - lead.x * dx;
                let r2 = rx * rx + ry * ry + lead.z * lead.z;
                let inv_r3 = area_gain / (r2 * r2.sqrt());
                let idx = flatten_2d_index(x, y, w);
                wx[idx] = rx * inv_r3;
                wy[idx] = ry * inv_r3;
            }
        }
    }
}
