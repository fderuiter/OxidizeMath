//! Unified spatial diffusion operator combining zero-drift `FiniteDifference2D` + `CellMask2D`
//! microfluidics with conservative upwind `AdvectionDiffusion2D` hydrodynamic transport.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_commons::math_kernel::types::{flatten_2d_index, Dimension, StepSize};
use math_explorer::biology::diffusion::{
    AdvectionDiffusion2D, CellMask2D, FiniteDifference2D, MaskedFiniteDifference2D,
    ReactionCellRole, SpatialDiffusion,
};
use math_explorer::biology::morphogenesis::ReactionKinetics;
use math_explorer::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

/// Spatial operator unifying microfluidic [`CellMask2D`] boundary roles with conservative
/// 2D upwind [`AdvectionDiffusion2D`] velocity and obstacle fields.
#[derive(Clone, Debug)]
pub struct CoupledMorphoDiffusion {
    /// Underlying Cartesian 5-point stencil operator for zero-drift stepping.
    pub base: FiniteDifference2D,
    /// Spatial cell-role mask (`ZeroFluxWall`, `EmitterU`, `EmitterV`, `Sink`).
    pub mask: CellMask2D,
    /// Pre-allocated conservative 2D upwind advection-diffusion operator.
    pub advection: AdvectionDiffusion2D,
    /// Whether hydrodynamic advection is active (`flow_mode != None`).
    pub flow_active: bool,
}

impl CoupledMorphoDiffusion {
    /// Constructs a pre-allocated [`CoupledMorphoDiffusion`] for a `width x height` lattice.
    #[must_use]
    pub fn new(width: Dimension, height: Dimension, dx: StepSize, dy: StepSize) -> Self {
        let w = *width;
        let h = *height;
        Self {
            base: FiniteDifference2D::new(width, height, dx, dy),
            mask: CellMask2D::new(w, h),
            advection: AdvectionDiffusion2D::new(width, height, dx, dy),
            flow_active: false,
        }
    }

    /// Returns the row-major solid obstacle slice (`true` = impermeable cell).
    #[must_use]
    pub fn obstacles(&self) -> &[bool] {
        self.advection.obstacles()
    }

    /// Synchronizes [`AdvectionDiffusion2D`] solid obstacles from [`ReactionCellRole::ZeroFluxWall`] cells in `mask`.
    pub fn sync_obstacles_from_mask(&mut self) {
        let roles = self.mask.roles();
        let (vx, vy, obs) = self.advection.buffers_mut();
        let count = obs.len().min(roles.len());
        for idx in 0..count {
            let is_wall = roles[idx] == ReactionCellRole::ZeroFluxWall as u8;
            obs[idx] = is_wall;
            if is_wall {
                vx[idx] = 0.0;
                vy[idx] = 0.0;
            }
        }
    }

    /// Paints or erases a circular obstacle disk in both `advection` and `mask`.
    pub fn paint_obstacle_disk(&mut self, cx: usize, cy: usize, radius: usize, is_obstacle: bool) {
        self.advection
            .paint_obstacle_disk(cx, cy, radius, is_obstacle);
        let role = if is_obstacle {
            ReactionCellRole::ZeroFluxWall
        } else {
            ReactionCellRole::Free
        };
        self.mask
            .paint_disk(cx as i32, cy as i32, radius as i32, role);
    }

    /// Clears all obstacles and cell-role masks across the grid.
    pub fn clear_all_obstacles(&mut self) {
        self.advection.clear_obstacles();
        self.mask.clear();
    }

    #[inline]
    fn neighbor_val(&self, slice: &[f64], center_val: f64, nb_idx: usize) -> f64 {
        if self.mask.roles()[nb_idx] == ReactionCellRole::ZeroFluxWall as u8 {
            center_val
        } else {
            slice[nb_idx]
        }
    }

    fn step_masked_zero_drift<const N: usize, K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        mut next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        let w = self.mask.width();
        let h = self.mask.height();
        if N == 0 || w == 0 || h == 0 {
            return;
        }
        let dx = *self.advection.dx();
        let dy = *self.advection.dy();
        let inv_dx_sq = 1.0 / (dx * dx);
        let inv_dy_sq = 1.0 / (dy * dy);
        for y in 0..h {
            let y_u = y.saturating_sub(1);
            let y_d = (y + 1).min(h - 1);
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                self.advance_masked_cell(
                    (x, y, y_u, y_d, w, idx),
                    (inv_dx_sq, inv_dy_sq, dt),
                    state,
                    &mut next_state,
                    coeffs,
                    kinetics,
                );
            }
        }
    }

    #[inline]
    fn advance_masked_cell<const N: usize, K: ReactionKinetics<N>>(
        &self,
        coords: (usize, usize, usize, usize, usize, usize),
        scales: (f64, f64, f64),
        state: [&[f64]; N],
        next_state: &mut [&mut [f64]; N],
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        let (x, y, y_u, y_d, w, idx) = coords;
        let (inv_dx_sq, inv_dy_sq, dt) = scales;
        match ReactionCellRole::from_u8(self.mask.roles()[idx]) {
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
                let idx_l = flatten_2d_index(x.saturating_sub(1), y, w);
                let idx_r = flatten_2d_index((x + 1).min(w - 1), y, w);
                let idx_u = flatten_2d_index(x, y_u, w);
                let idx_d = flatten_2d_index(x, y_d, w);
                for s in 0..N {
                    let u_c = curr[s];
                    let u_l = self.neighbor_val(state[s], u_c, idx_l);
                    let u_r = self.neighbor_val(state[s], u_c, idx_r);
                    let u_u = self.neighbor_val(state[s], u_c, idx_u);
                    let u_d = self.neighbor_val(state[s], u_c, idx_d);
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

impl<const N: usize> SpatialDiffusion<N> for CoupledMorphoDiffusion {
    fn stepper(&self) -> FusedStencilStepper {
        if self.flow_active {
            <AdvectionDiffusion2D as SpatialDiffusion<N>>::stepper(&self.advection)
        } else {
            <FiniteDifference2D as SpatialDiffusion<N>>::stepper(&self.base)
        }
    }

    fn step_fused<K: ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        mut next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        if self.flow_active {
            self.advection.step_fused(
                state,
                next_state.each_mut().map(|s| &mut **s),
                dt,
                coeffs,
                kinetics,
            );
            if self.mask.has_active_cells() {
                let roles = self.mask.roles();
                for (idx, &code) in roles.iter().enumerate() {
                    match ReactionCellRole::from_u8(code) {
                        ReactionCellRole::ZeroFluxWall | ReactionCellRole::Sink => {
                            for next_s in next_state.iter_mut().take(N) {
                                next_s[idx] = 0.0;
                            }
                        }
                        ReactionCellRole::EmitterU if N > 0 => {
                            next_state[0][idx] = self.mask.emitter_u_level;
                        }
                        ReactionCellRole::EmitterV if N > 1 => {
                            next_state[1][idx] = self.mask.emitter_v_level;
                        }
                        _ => {}
                    }
                }
            }
        } else if self.mask.has_active_cells() {
            self.step_masked_zero_drift(state, next_state, dt, coeffs, kinetics);
        } else {
            self.base
                .step_fused(state, next_state, dt, coeffs, kinetics);
        }
    }

    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        if self.flow_active {
            self.advection.map_diffusion(state, coeffs, op);
        } else if self.mask.has_active_cells() {
            let masked = MaskedFiniteDifference2D {
                base: self.base,
                mask: self.mask.clone(),
            };
            masked.map_diffusion(state, coeffs, op);
        } else {
            self.base.map_diffusion(state, coeffs, op);
        }
    }
}
