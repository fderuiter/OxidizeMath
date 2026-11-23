//! Spatial cell-role masks for 2D reaction-diffusion microfluidics:
//! internal zero-flux walls, persistent `u`/`v` chemical emitters, and sinks.

use super::{FiniteDifference2D, SpatialDiffusion};
use crate::biology::diffusion::fd2d::geometry::GeometryStrategy;
use crate::biology::morphogenesis::reaction::ReactionKinetics;
use crate::biology::reaction_diffusion::{ChemicalState, DiffusionModel};
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

/// Discrete boundary/source role assigned to a 2D reaction-diffusion lattice cell.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReactionCellRole {
    /// Standard interior fluid cell evolving under coupled reaction-diffusion PDEs.
    #[default]
    Free = 0,
    /// Impermeable wall enforcing internal zero-flux ($\nabla u \cdot \mathbf{n} = 0$) reflection.
    ZeroFluxWall = 1,
    /// Persistent chemical source injecting activator species `u` (species index `0`).
    EmitterU = 2,
    /// Persistent chemical source injecting inhibitor species `v` (species index `1`).
    EmitterV = 3,
    /// Chemical absorber clamping local species concentrations to zero.
    Sink = 4,
}

impl ReactionCellRole {
    /// Converts a compact byte code into a [`ReactionCellRole`].
    #[verified_engine::verified]
    pub fn from_u8(code: u8) -> Self {
        match code {
            1 => Self::ZeroFluxWall,
            2 => Self::EmitterU,
            3 => Self::EmitterV,
            4 => Self::Sink,
            _ => Self::Free,
        }
    }
}

/// Pre-allocated 2D spatial mask storing per-cell [`ReactionCellRole`] values.
#[derive(Debug, Clone)]
pub struct CellMask2D {
    width: usize,
    height: usize,
    roles: Vec<u8>,
    has_active: bool,
    /// Pinned activator concentration at [`ReactionCellRole::EmitterU`] cells.
    pub emitter_u_level: f64,
    /// Pinned inhibitor concentration at [`ReactionCellRole::EmitterV`] cells.
    pub emitter_v_level: f64,
}

impl CellMask2D {
    /// Allocates an all-`Free` 2D cell mask for a `width x height` grid.
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize) -> Self {
        let len = width.checked_mul(height).expect("Mask grid size overflow");
        Self {
            width,
            height,
            roles: vec![ReactionCellRole::Free as u8; len],
            has_active: false,
            emitter_u_level: 1.8,
            emitter_v_level: 1.8,
        }
    }

    /// Returns the grid width.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the grid height.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the raw row-major `u8` role buffer.
    #[verified_engine::verified]
    pub fn roles(&self) -> &[u8] {
        &self.roles
    }

    /// Returns `true` when at least one cell is non-`Free`.
    #[verified_engine::verified]
    pub fn has_active_cells(&self) -> bool {
        self.has_active
    }

    /// Reads the [`ReactionCellRole`] at `(x, y)`.
    #[verified_engine::verified]
    pub fn get(&self, x: usize, y: usize) -> ReactionCellRole {
        if x < self.width && y < self.height {
            ReactionCellRole::from_u8(self.roles[y * self.width + x])
        } else {
            ReactionCellRole::Free
        }
    }

    /// Sets the [`ReactionCellRole`] at `(x, y)`.
    #[verified_engine::verified]
    pub fn set(&mut self, x: usize, y: usize, role: ReactionCellRole) {
        if x < self.width && y < self.height {
            self.roles[y * self.width + x] = role as u8;
            if role != ReactionCellRole::Free {
                self.has_active = true;
            } else {
                self.recompute_active_flag();
            }
        }
    }

    /// Paints a filled circular disk of `role` centered at `(cx, cy)`.
    #[verified_engine::verified]
    pub fn paint_disk(&mut self, cx: i32, cy: i32, radius: i32, role: ReactionCellRole) {
        let r = radius.max(0);
        let r_sq = r * r;
        let w = self.width as i32;
        let h = self.height as i32;
        for y in (cy - r)..=(cy + r) {
            for x in (cx - r)..=(cx + r) {
                if x >= 0 && x < w && y >= 0 && y < h && (x - cx).pow(2) + (y - cy).pow(2) <= r_sq {
                    self.roles[y as usize * self.width + x as usize] = role as u8;
                }
            }
        }
        self.recompute_active_flag();
    }

    /// Resets every cell in the mask to [`ReactionCellRole::Free`].
    #[verified_engine::verified]
    pub fn clear(&mut self) {
        self.roles.fill(ReactionCellRole::Free as u8);
        self.has_active = false;
    }

    /// Replaces the mask buffer from a validated byte slice without reallocating.
    #[verified_engine::verified]
    pub fn set_roles_slice(&mut self, bytes: &[u8]) -> Result<(), &'static str> {
        if bytes.len() != self.roles.len() {
            return Err("CellMask2D byte slice length mismatch");
        }
        for (dst, &code) in self.roles.iter_mut().zip(bytes.iter()) {
            *dst = ReactionCellRole::from_u8(code) as u8;
        }
        self.recompute_active_flag();
        Ok(())
    }

    /// Stamps a microfluidic chicane maze with zero-flux baffles, upstream `u`/`v`
    /// emitters, and a downstream chemical sink.
    #[verified_engine::verified]
    pub fn stamp_microfluidic_maze(&mut self) {
        self.clear();
        let w = self.width;
        let h = self.height;
        if w < 12 || h < 12 {
            return;
        }
        let wall = ReactionCellRole::ZeroFluxWall;
        let y1 = h / 3;
        let y2 = (2 * h) / 3;
        for dy in 0..2 {
            let r1 = (y1 + dy).min(h - 1);
            for x in 0..((3 * w) / 4) {
                self.set(x, r1, wall);
            }
            let r2 = (y2 + dy).min(h - 1);
            for x in (w / 4)..w {
                self.set(x, r2, wall);
            }
        }
        self.paint_disk(
            (w / 4) as i32,
            (h / 6) as i32,
            3,
            ReactionCellRole::EmitterU,
        );
        self.paint_disk(
            (w / 2) as i32,
            (h / 6) as i32,
            3,
            ReactionCellRole::EmitterV,
        );
        self.paint_disk(
            ((3 * w) / 4) as i32,
            ((5 * h) / 6) as i32,
            4,
            ReactionCellRole::Sink,
        );
    }

    #[verified_engine::verified]
    fn recompute_active_flag(&mut self) {
        self.has_active = self
            .roles
            .iter()
            .any(|&b| b != ReactionCellRole::Free as u8);
    }
}

/// Masked 2D finite-difference operator enforcing internal zero-flux walls,
/// persistent `u`/`v` emitters, and sinks with zero heap allocations during stepping.
#[derive(Debug, Clone)]
pub struct MaskedFiniteDifference2D {
    /// Underlying Cartesian 5-point stencil operator.
    pub base: FiniteDifference2D,
    /// Spatial cell-role mask.
    pub mask: CellMask2D,
}

impl MaskedFiniteDifference2D {
    /// Creates a masked 2D finite-difference operator matching `base`'s grid dimensions.
    #[verified_engine::verified]
    pub fn new(base: FiniteDifference2D) -> Self {
        let w = *base.geometry.width;
        let h = *base.geometry.height;
        Self {
            base,
            mask: CellMask2D::new(w, h),
        }
    }

    #[inline]
    #[verified_engine::verified]
    fn neighbor_val(&self, slice: &[f64], center_val: f64, nb_idx: usize) -> f64 {
        if self.mask.roles[nb_idx] == ReactionCellRole::ZeroFluxWall as u8 {
            center_val
        } else {
            slice[nb_idx]
        }
    }
}

impl DiffusionModel for MaskedFiniteDifference2D {
    #[verified_engine::verified]
    fn apply(&self, state: &ChemicalState, out: &mut ChemicalState, coeffs: &[f64]) {
        if !self.mask.has_active {
            DiffusionModel::apply(&self.base, state, out, coeffs);
            return;
        }
        let w = *self.base.geometry.width;
        let h = *self.base.geometry.height;
        let inv_dx_sq = 1.0 / (*self.base.geometry.dx() * *self.base.geometry.dx());
        let inv_dy_sq = 1.0 / (*self.base.geometry.dy() * *self.base.geometry.dy());

        for (s, &coeff) in coeffs.iter().enumerate().take(state.num_species()) {
            let src = state.species(s);
            let dst = out.species_mut(s);
            let cx = coeff * inv_dx_sq;
            let cy = coeff * inv_dy_sq;
            for y in 0..h {
                let y_u = y.saturating_sub(1);
                let y_d = (y + 1).min(h - 1);
                for x in 0..w {
                    let idx = y * w + x;
                    if self.mask.roles[idx] == ReactionCellRole::ZeroFluxWall as u8 {
                        dst[idx] = 0.0;
                        continue;
                    }
                    let u_c = src[idx];
                    let u_l = self.neighbor_val(src, u_c, y * w + x.saturating_sub(1));
                    let u_r = self.neighbor_val(src, u_c, y * w + (x + 1).min(w - 1));
                    let u_u = self.neighbor_val(src, u_c, y_u * w + x);
                    let u_d = self.neighbor_val(src, u_c, y_d * w + x);
                    dst[idx] = (u_l + u_r - 2.0 * u_c) * cx + (u_u + u_d - 2.0 * u_c) * cy;
                }
            }
        }
    }
}

impl<const N: usize> SpatialDiffusion<N> for MaskedFiniteDifference2D {
    fn stepper(&self) -> FusedStencilStepper {
        <FiniteDifference2D as SpatialDiffusion<N>>::stepper(&self.base)
    }

    #[verified_engine::verified]
    fn step_fused<K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        mut next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        if !self.mask.has_active {
            self.base
                .step_fused(state, next_state, dt, coeffs, kinetics);
            return;
        }
        let w = *self.base.geometry.width;
        let h = *self.base.geometry.height;
        if N == 0 || w == 0 || h == 0 {
            return;
        }
        let inv_dx_sq = 1.0 / (*self.base.geometry.dx() * *self.base.geometry.dx());
        let inv_dy_sq = 1.0 / (*self.base.geometry.dy() * *self.base.geometry.dy());

        for y in 0..h {
            let y_u = y.saturating_sub(1);
            let y_d = (y + 1).min(h - 1);
            for x in 0..w {
                let idx = y * w + x;
                let role = ReactionCellRole::from_u8(self.mask.roles[idx]);
                match role {
                    ReactionCellRole::ZeroFluxWall | ReactionCellRole::Sink => {
                        for next_s in next_state.iter_mut().take(N) {
                            next_s[idx] = 0.0;
                        }
                    }
                    ReactionCellRole::EmitterU => {
                        next_state[0][idx] = self.mask.emitter_u_level;
                        for s in 1..N {
                            next_state[s][idx] = state[s][idx].clamp(0.0, 2.5);
                        }
                    }
                    ReactionCellRole::EmitterV => {
                        if N > 1 {
                            next_state[0][idx] = state[0][idx].clamp(0.0, 2.5);
                            next_state[1][idx] = self.mask.emitter_v_level;
                            for s in 2..N {
                                next_state[s][idx] = state[s][idx].clamp(0.0, 2.5);
                            }
                        } else {
                            next_state[0][idx] = self.mask.emitter_v_level;
                        }
                    }
                    ReactionCellRole::Free => {
                        let mut curr = [0.0; N];
                        for s in 0..N {
                            curr[s] = state[s][idx];
                        }
                        let rates = kinetics.reaction(curr);
                        for s in 0..N {
                            let u_c = curr[s];
                            let u_l = self.neighbor_val(state[s], u_c, y * w + x.saturating_sub(1));
                            let u_r = self.neighbor_val(state[s], u_c, y * w + (x + 1).min(w - 1));
                            let u_u = self.neighbor_val(state[s], u_c, y_u * w + x);
                            let u_d = self.neighbor_val(state[s], u_c, y_d * w + x);
                            let d_s = coeffs[s].max(0.0);
                            let nb_flux = d_s * ((u_l + u_r) * inv_dx_sq + (u_u + u_d) * inv_dy_sq);
                            let alpha = 2.0 * dt * d_s * (inv_dx_sq + inv_dy_sq);
                            let rate = rates[s];
                            next_state[s][idx] = if rate < 0.0 && u_c > 1e-12 {
                                (u_c + dt * nb_flux) / (1.0 + alpha + dt * (-rate / u_c))
                            } else {
                                (u_c + dt * (nb_flux + rate)) / (1.0 + alpha)
                            };
                        }
                    }
                }
            }
        }
    }

    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], mut op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        if !self.mask.has_active {
            self.base.map_diffusion(state, coeffs, op);
            return;
        }
        let w = *self.base.geometry.width;
        let h = *self.base.geometry.height;
        let inv_dx_sq = 1.0 / (*self.base.geometry.dx() * *self.base.geometry.dx());
        let inv_dy_sq = 1.0 / (*self.base.geometry.dy() * *self.base.geometry.dy());
        for y in 0..h {
            let y_u = y.saturating_sub(1);
            let y_d = (y + 1).min(h - 1);
            for x in 0..w {
                let idx = y * w + x;
                let mut curr = [0.0; N];
                let mut diffs = [0.0; N];
                if self.mask.roles[idx] != ReactionCellRole::ZeroFluxWall as u8 {
                    for s in 0..N {
                        let u_c = state[s][idx];
                        let u_l = self.neighbor_val(state[s], u_c, y * w + x.saturating_sub(1));
                        let u_r = self.neighbor_val(state[s], u_c, y * w + (x + 1).min(w - 1));
                        let u_u = self.neighbor_val(state[s], u_c, y_u * w + x);
                        let u_d = self.neighbor_val(state[s], u_c, y_d * w + x);
                        curr[s] = u_c;
                        diffs[s] = coeffs[s]
                            * ((u_l + u_r - 2.0 * u_c) * inv_dx_sq
                                + (u_u + u_d - 2.0 * u_c) * inv_dy_sq);
                    }
                }
                op(idx, curr, diffs);
            }
        }
    }
}
