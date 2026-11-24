use super::SpatialDiffusion;
use math_commons::math_kernel::types::StepSize;

/// A 1D Finite Difference implementation using a 3-point stencil.
///
/// Handles boundaries with Neumann conditions (zero flux).
///
/// # The Stencil
///
/// ```mermaid
/// graph LR
///     L[u_{i-1}] --> C[u_i]
///     R[u_{i+1}] --> C
///     style C fill:#f9f,stroke:#333,stroke-width:2px
/// ```
///
/// $$ \nabla^2 u \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{dx^2} $$
#[derive(Debug, Clone, Copy)]
pub struct FiniteDifference1D {
    /// Grid spacing.
    pub dx: StepSize,
}

impl FiniteDifference1D {
    /// Creates a new 1D finite difference strategy.
    #[verified_engine::verified]
    pub fn new(dx: StepSize) -> Self {
        Self { dx }
    }
}

impl crate::biology::reaction_diffusion::DiffusionModel for FiniteDifference1D {
    #[verified_engine::verified]
    fn apply(
        &self,
        state: &crate::biology::reaction_diffusion::ChemicalState,
        out: &mut crate::biology::reaction_diffusion::ChemicalState,
        coeffs: &[f64],
    ) {
        let n_species = state.num_species();
        let dx_sq = *self.dx * *self.dx;
        let inv_dx_sq = 1.0 / dx_sq;

        for (s, d) in coeffs.iter().enumerate().take(n_species) {
            let u = state.species(s);
            let out_u = out.species_mut(s);
            let d = *d;
            apply_1d_stencil(u, out_u, d, inv_dx_sq);
        }
    }
}

use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

impl<const N: usize> SpatialDiffusion<N> for FiniteDifference1D {
    fn stepper(&self) -> FusedStencilStepper {
        FusedStencilStepper::new(self.dx)
    }

    #[verified_engine::verified]
    fn step_fused<K: crate::biology::morphogenesis::reaction::ReactionKinetics<N>>(
        &self,
        state: [&[f64]; N],
        next_state: [&mut [f64]; N],
        dt: f64,
        coeffs: [f64; N],
        kinetics: &K,
    ) {
        if N == 0 {
            return;
        }
        let n = state[0].len();
        if n == 0 {
            return;
        }

        SpatialDiffusion::<N>::stepper(self).step_1d_coupled_neumann(
            n,
            state,
            next_state,
            dt,
            1.0, // Forward time
            |i, _prev, curr, _next, ops| {
                let mut rhs = [0.0; N];
                let rates = kinetics.reaction_at(i, curr);
                for s in 0..N {
                    let d2u = ops.central_diff_2nd(_prev[s], curr[s], _next[s]);
                    rhs[s] = coeffs[s] * d2u + rates[s];
                }
                rhs
            },
        );
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    #[verified_engine::verified]
    fn map_diffusion<F>(&self, state: [&[f64]; N], coeffs: [f64; N], mut op: F)
    where
        F: FnMut(usize, [f64; N], [f64; N]),
    {
        if N == 0 {
            return;
        }
        let n = state[0].len();
        if n == 0 {
            return;
        }

        let dx_sq = *self.dx * *self.dx;
        let inv_dx_sq = 1.0 / dx_sq;

        // Precompute coefficients
        let mut d_inv_dx_sq = [0.0; N];
        for s in 0..N {
            d_inv_dx_sq[s] = coeffs[s] * inv_dx_sq;
        }

        // Verify all buffers are large enough
        for (s, buffer) in state.iter().enumerate().take(N) {
            assert!(
                buffer.len() >= n,
                "Buffer too small for diffusion (species {})",
                s
            );
        }

        // 1. Left Boundary (i=0)
        {
            let i = 0;
            let mut current_vals = [0.0; N];
            let mut diff_vals = [0.0; N];

            for s in 0..N {
                let u = state[s];
                // Safety: checked n > 0 at start of function
                let u_curr = u[0];
                let u_prev = u_curr; // Neumann: u_{-1} = u_0
                let u_next = if n > 1 { u[1] } else { u_curr };

                let lap = (u_next - 2.0 * u_curr + u_prev) * d_inv_dx_sq[s];
                current_vals[s] = u_curr;
                diff_vals[s] = lap;
            }
            op(i, current_vals, diff_vals);
        }

        // 2. Interior (Optimized loop)
        if n > 2 {
            for i in 1..n - 1 {
                let mut current_vals = [0.0; N];
                let mut diff_vals = [0.0; N];

                for s in 0..N {
                    let u = state[s];
                    // SAFETY: i is in 1..n-1, so i-1, i, i+1 are in 0..n.
                    // We asserted buffer.len() >= n above.
                    // Transitioning to safe indexing.
                    let u_curr = u[i];
                    let u_prev = u[i - 1];
                    let u_next = u[i + 1];

                    let lap = (u_next - 2.0 * u_curr + u_prev) * d_inv_dx_sq[s];
                    current_vals[s] = u_curr;
                    diff_vals[s] = lap;
                }
                op(i, current_vals, diff_vals);
            }
        }

        // 3. Right Boundary (i=n-1)
        if n > 1 {
            let i = n - 1;
            let mut current_vals = [0.0; N];
            let mut diff_vals = [0.0; N];

            for s in 0..N {
                let u = state[s];
                // Safety: i = n-1, checked n > 1
                let u_curr = u[i];
                let u_prev = u[i - 1];
                let u_next = u_curr; // Neumann: u_{N} = u_{N-1}

                let lap = (u_next - 2.0 * u_curr + u_prev) * d_inv_dx_sq[s];
                current_vals[s] = u_curr;
                diff_vals[s] = lap;
            }
            op(i, current_vals, diff_vals);
        }
    }
}

/// Applies a 1D Finite Difference stencil (Neumann BC) to a single array.
///
/// This is a private helper to ensure DRY compliance between DiffusionModel and SpatialDiffusion implementations.
///
/// # Arguments
/// * `src`: Input concentration slice.
/// * `dst`: Output buffer for the Laplacian term (D * d2u/dx2).
/// * `d`: Diffusion coefficient.
/// * `inv_dx_sq`: Inverse square of grid spacing (1/dx^2).
#[verified_engine::verified]
fn apply_1d_stencil(src: &[f64], dst: &mut [f64], d: f64, inv_dx_sq: f64) {
    scan_1d_stencil(src, d, inv_dx_sq, |i, val| {
        // Safety: We rely on the caller to ensure dst is sized correctly.
        // scan_1d_stencil guarantees i < src.len(), which should match dst.len().
        if i < dst.len() {
            dst[i] = val;
        }
    });
}

/// Helper to apply 1D Finite Difference stencil.
/// Calls `op` with (index, laplacian_value) for each point.
#[verified_engine::verified]
fn scan_1d_stencil<F>(src: &[f64], d: f64, inv_dx_sq: f64, mut op: F)
where
    F: FnMut(usize, f64),
{
    let n = src.len();
    if n == 0 {
        return;
    }

    // 1. Left Boundary (i=0)
    {
        let u_curr = src[0];
        let u_prev = u_curr; // Neumann: u_{-1} = u_0
        let u_next = if n > 1 { src[1] } else { u_curr };
        let lap = d * (u_next - 2.0 * u_curr + u_prev) * inv_dx_sq;
        op(0, lap);
    }

    // 2. Interior (Optimized with windows iterator)
    if n > 2 {
        // Iterate over windows of 3 elements: [prev, curr, next]
        // Window index 0 corresponds to center index 1.
        for (i, win) in src.windows(3).enumerate() {
            let u_prev = win[0];
            let u_curr = win[1];
            let u_next = win[2];
            let lap = d * (u_next - 2.0 * u_curr + u_prev) * inv_dx_sq;
            op(i + 1, lap);
        }
    }

    // 3. Right Boundary (i=n-1)
    if n > 1 {
        let i = n - 1;
        let u_curr = src[i];
        let u_prev = src[i - 1];
        let u_next = u_curr; // Neumann: u_{N} = u_{N-1}
        let lap = d * (u_next - 2.0 * u_curr + u_prev) * inv_dx_sq;
        op(i, lap);
    }
}
