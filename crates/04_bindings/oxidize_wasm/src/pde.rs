//! Zero-allocation `#[wasm_bindgen]` continuum PDE & radix-2 FFT workspace wrapping `engine_pde`.
// [cite:engine_pde]

use crate::error::WasmSdkError;
use engine_pde::{
    FusedStencilStepper, OwnedByteArena, PdeError, StencilNeighbors3D, StencilOperators,
    try_fft_1d_inplace, try_fft_2d_inplace,
};
use math_commons::math_kernel::types::{StepSize, flatten_3d_index};
use wasm_bindgen::prelude::*;

/// Maximum total grid cells supported by a single `WasmPdeWorkspace` (`2,097,152` = `128^3`).
pub const MAX_PDE_CELLS: usize = 2_097_152;

/// Reusable zero-allocation 1D/2D/3D finite-difference PDE & radix-2 FFT workspace.
///
/// Maintains ping-pong species buffers (`u`, `v`) directly inside WASM linear memory so JavaScript
/// and Web Worker callers can step coupled reaction-diffusion systems and read [`Self::u_ptr`] /
/// [`Self::v_ptr`] via zero-copy `Float64Array` views.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmPdeWorkspace {
    dims: (usize, usize, usize),
    stepper: FusedStencilStepper,
    u_curr: Vec<f64>,
    v_curr: Vec<f64>,
    u_next: Vec<f64>,
    v_next: Vec<f64>,
    col_scratch: Vec<f64>,
    scratch: OwnedByteArena,
}

#[wasm_bindgen]
impl WasmPdeWorkspace {
    /// Pre-allocates a 2D coupled PDE & FFT workspace of shape `width x height`.
    #[wasm_bindgen(js_name = tryNew2d)]
    pub fn try_new_2d(width: usize, height: usize, dx: f64, dy: f64) -> Result<Self, WasmSdkError> {
        Self::try_new_3d(width, height, 1, dx, dy, 1.0)
    }

    /// Pre-allocates a 3D coupled PDE workspace of shape `nx x ny x nz` (`nx * ny * nz <= 128^3`).
    #[wasm_bindgen(js_name = tryNew3d)]
    pub fn try_new_3d(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Result<Self, WasmSdkError> {
        let cells = nx
            .checked_mul(ny)
            .and_then(|xy| xy.checked_mul(nz))
            .unwrap_or(usize::MAX);
        if nx == 0 || ny == 0 || nz == 0 || cells > MAX_PDE_CELLS {
            return Err(WasmSdkError::invalid_argument(
                "E_PDE_GRID_SIZE",
                format!("Grid ({nx}x{ny}x{nz}) cell count must be within 1..={MAX_PDE_CELLS}"),
                MAX_PDE_CELLS,
                cells,
            ));
        }
        let sx = StepSize(dx);
        let sy = StepSize(dy);
        let sz = StepSize(dz);
        let stepper = if nz == 1 && ny == 1 {
            FusedStencilStepper::try_new(sx)?
        } else if nz == 1 {
            FusedStencilStepper::try_new_2d(sx, sy)?
        } else {
            FusedStencilStepper::try_new_3d(sx, sy, sz)?
        };
        let scratch_bytes = cells.min(65_536).saturating_mul(std::mem::size_of::<f64>());
        let scratch = OwnedByteArena::try_with_capacity(scratch_bytes)?;
        let col_cap = ny.max(nx).saturating_mul(2);
        Ok(Self {
            dims: (nx, ny, nz),
            stepper,
            u_curr: vec![0.0; cells],
            v_curr: vec![0.0; cells],
            u_next: vec![0.0; cells],
            v_next: vec![0.0; cells],
            col_scratch: vec![0.0; col_cap],
            scratch,
        })
    }

    /// Total number of cells (`nx * ny * nz`) in each species grid.
    #[wasm_bindgen(js_name = gridLen)]
    #[must_use]
    pub fn grid_len(&self) -> usize {
        self.u_curr.len()
    }

    /// Byte offset of activator field `u` in WASM linear memory (`Float64Array` view).
    #[wasm_bindgen(js_name = uPtr)]
    #[must_use]
    pub fn u_ptr(&self) -> usize {
        self.u_curr.as_ptr() as usize
    }

    /// Byte offset of inhibitor field `v` in WASM linear memory (`Float64Array` view).
    #[wasm_bindgen(js_name = vPtr)]
    #[must_use]
    pub fn v_ptr(&self) -> usize {
        self.v_curr.as_ptr() as usize
    }

    /// Capacity of pre-allocated scratch workspace memory.
    #[wasm_bindgen(js_name = scratchCapacity)]
    #[must_use]
    pub fn scratch_capacity(&mut self) -> usize {
        self.scratch.byte_capacity() / std::mem::size_of::<f64>()
    }

    /// Deterministic steady-state + sinusoidal perturbation seeder for Schnakenberg morphogenesis.
    #[wasm_bindgen(js_name = seedSchnakenbergSteadyState)]
    pub fn seed_schnakenberg_steady_state(&mut self, a: f64, b: f64, amp: f64) {
        let u0 = (a + b).max(1e-6);
        let v0 = b / (u0 * u0);
        let (nx, ny, nz) = self.dims;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = flatten_3d_index(x, y, z, nx, ny);
                    let phase = (0.65 * (x as f64)).sin()
                        * (0.65 * (y as f64)).cos()
                        * (0.45 * (z as f64)).cos();
                    self.u_curr[idx] = (u0 + amp * phase).max(0.0);
                    self.v_curr[idx] = (v0 - 0.5 * amp * phase).max(0.0);
                }
            }
        }
    }

    /// Copies caller `u` and `v` slices into the pre-allocated grid buffers.
    #[wasm_bindgen(js_name = loadSpecies)]
    pub fn load_species(&mut self, u: &[f64], v: &[f64]) -> Result<(), WasmSdkError> {
        let n = self.u_curr.len();
        if u.len() != n || v.len() != n {
            return Err(WasmSdkError::invalid_argument(
                "E_PDE_SLICE_LEN",
                "Species slices must match gridLen()",
                n,
                u.len().min(v.len()),
            ));
        }
        self.u_curr.copy_from_slice(u);
        self.v_curr.copy_from_slice(v);
        Ok(())
    }

    /// Advances 2D zero-flux Neumann Schnakenberg reaction-diffusion in place for `steps` ticks.
    #[wasm_bindgen(js_name = stepSchnakenberg2dInPlace)]
    pub fn step_schnakenberg_2d_in_place(
        &mut self,
        dt: f64,
        du: f64,
        dv: f64,
        a: f64,
        b: f64,
        steps: usize,
    ) -> Result<(), WasmSdkError> {
        if !(dt > 0.0 && dt.is_finite() && du >= 0.0 && dv >= 0.0) {
            return Err(PdeError::InvalidParameter { parameter: "dt" }.into());
        }
        let (nx, ny, nz) = self.dims;
        if nz != 1 {
            return Err(PdeError::InvalidGridDimension {
                axis: "z",
                size: nz,
            }
            .into());
        }
        for _ in 0..steps.max(1) {
            self.stepper.step_2d_coupled_neumann(
                (nx, ny),
                [&self.u_curr, &self.v_curr],
                [&mut self.u_next, &mut self.v_next],
                dt,
                1.0,
                |_idx, c, l, r, up, dn, ops: &StencilOperators| {
                    let lap_u = ops.central_diff_2nd_2d(c[0], l[0], r[0], up[0], dn[0]);
                    let lap_v = ops.central_diff_2nd_2d(c[1], l[1], r[1], up[1], dn[1]);
                    let u2v = c[0] * c[0] * c[1];
                    [du * lap_u + a - c[0] + u2v, dv * lap_v + b - u2v]
                },
            );
            std::mem::swap(&mut self.u_curr, &mut self.u_next);
            std::mem::swap(&mut self.v_curr, &mut self.v_next);
        }
        Ok(())
    }

    /// Advances 3D zero-flux Neumann Schnakenberg reaction-diffusion in place for `steps` ticks.
    #[wasm_bindgen(js_name = stepSchnakenberg3dInPlace)]
    pub fn step_schnakenberg_3d_in_place(
        &mut self,
        dt: f64,
        du: f64,
        dv: f64,
        a: f64,
        b: f64,
        steps: usize,
    ) -> Result<(), WasmSdkError> {
        if !(dt > 0.0 && dt.is_finite() && du >= 0.0 && dv >= 0.0) {
            return Err(PdeError::InvalidParameter { parameter: "dt" }.into());
        }
        let (nx, ny, nz) = self.dims;
        for _ in 0..steps.max(1) {
            self.stepper.step_3d_coupled_neumann(
                (nx, ny, nz),
                [&self.u_curr, &self.v_curr],
                [&mut self.u_next, &mut self.v_next],
                dt,
                1.0,
                |_idx, curr, nb: StencilNeighbors3D<2>, ops: &StencilOperators| {
                    let lap_u = ops.central_diff_2nd_3d(
                        curr[0],
                        [nb.xm[0], nb.xp[0], nb.ym[0], nb.yp[0], nb.zm[0], nb.zp[0]],
                    );
                    let lap_v = ops.central_diff_2nd_3d(
                        curr[1],
                        [nb.xm[1], nb.xp[1], nb.ym[1], nb.yp[1], nb.zm[1], nb.zp[1]],
                    );
                    let u2v = curr[0] * curr[0] * curr[1];
                    [du * lap_u + a - curr[0] + u2v, dv * lap_v + b - u2v]
                },
            );
            std::mem::swap(&mut self.u_curr, &mut self.u_next);
            std::mem::swap(&mut self.v_curr, &mut self.v_next);
        }
        Ok(())
    }

    /// Executes a zero-allocation in-place radix-2 1D Cooley-Tukey FFT on interleaved `[re_0, im_0, ...]`.
    #[wasm_bindgen(js_name = fft1dInterleaved)]
    pub fn fft_1d_interleaved(
        &mut self,
        interleaved: &mut [f64],
        inverse: bool,
    ) -> Result<(), WasmSdkError> {
        if interleaved.len() < 2 || !interleaved.len().is_multiple_of(2) {
            return Err(WasmSdkError::invalid_argument(
                "E_FFT_INTERLEAVED_LEN",
                "Interleaved complex buffer length must be an even positive integer",
                2,
                interleaved.len(),
            ));
        }
        let n = interleaved.len() / 2;
        try_fft_1d_inplace(interleaved, n, inverse)?;
        Ok(())
    }

    /// Executes a zero-allocation in-place radix-2 2D FFT on `(nx, ny)` interleaved `[re, im, ...]` data.
    #[wasm_bindgen(js_name = fft2dInterleaved)]
    pub fn fft_2d_interleaved(
        &mut self,
        nx: usize,
        ny: usize,
        interleaved: &mut [f64],
        inverse: bool,
    ) -> Result<(), WasmSdkError> {
        let req_col = ny.saturating_mul(2);
        if self.col_scratch.len() < req_col {
            self.col_scratch.resize(req_col, 0.0);
        }
        try_fft_2d_inplace(
            interleaved,
            nx,
            ny,
            &mut self.col_scratch[..req_col],
            inverse,
        )?;
        Ok(())
    }
}

impl WasmPdeWorkspace {
    /// Immutable Rust slice view of activator field `u`.
    #[must_use]
    pub fn u_slice(&self) -> &[f64] {
        &self.u_curr
    }

    /// Immutable Rust slice view of inhibitor field `v`.
    #[must_use]
    pub fn v_slice(&self) -> &[f64] {
        &self.v_curr
    }
}
