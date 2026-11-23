//! Hydrodynamic force evaluation via bounce-back momentum exchange.

use super::model::{CollisionModel, Lattice2D, LatticeBoltzmann};

impl<const Q: usize, L: Lattice2D<Q>, C: CollisionModel<Q, L>> LatticeBoltzmann<Q, L, C> {
    /// Returns the hydrodynamic `(drag_force, lift_force)` accumulated during the latest
    /// bounce-back streaming step, or evaluates it directly from `self.state.f` if not yet stepped.
    #[verified_engine::verified]
    pub fn hydrodynamic_forces(&self) -> (f64, f64) {
        if self.state.has_stepped_forces {
            (self.state.last_drag_force, self.state.last_lift_force)
        } else {
            self.compute_bounce_back_forces()
        }
    }

    /// Computes bounce-back momentum-exchange forces `(F_D, F_L)` directly from the
    /// current post-collision distribution `f[(y, x)][k]` across all fluid-solid links:
    /// $F_D = \sum 2 f_{(y, x), k} c_{kx}$ and $F_L = \sum 2 f_{(y, x), k} c_{ky}$.
    #[verified_engine::verified]
    pub fn compute_bounce_back_forces(&self) -> (f64, f64) {
        let width = self.state.width;
        let height = self.state.height;
        let cx = L::directions_x();
        let cy = L::directions_y();
        let mut f_drag = 0.0;
        let mut f_lift = 0.0;

        for y in 0..height {
            for x in 0..width {
                if self.state.obstacles[(y, x)] {
                    continue;
                }
                for k in 0..Q {
                    if cx[k] == 0 && cy[k] == 0 {
                        continue;
                    }
                    let nx = x as isize + cx[k] as isize;
                    let ny = y as isize + cy[k] as isize;
                    if nx >= 0
                        && (nx as usize) < width
                        && ny >= 0
                        && (ny as usize) < height
                        && self.state.obstacles[(ny as usize, nx as usize)]
                    {
                        let two_f = 2.0 * self.state.f[(y, x)][k];
                        f_drag += two_f * (cx[k] as f64);
                        f_lift += two_f * (cy[k] as f64);
                    }
                }
            }
        }
        (f_drag, f_lift)
    }
}

/// Computes the dimensionless drag coefficient $C_D = \frac{F_D}{\frac{1}{2} \rho_0 U_0^2 D_{\text{char}}}$.
#[verified_engine::verified]
pub fn compute_drag_coefficient(
    drag_force: f64,
    rho_ref: f64,
    u_inlet: f64,
    char_length: f64,
) -> f64 {
    let denom = 0.5 * rho_ref * u_inlet * u_inlet * char_length;
    if denom <= math_commons::registry::TOLERANCE_HIGH || !denom.is_finite() {
        return 0.0;
    }
    drag_force / denom
}
