use super::model::{CollisionModel, Lattice2D, LatticeBoltzmann};
use super::state::LatticeState;

impl<const Q: usize> LatticeState<Q> {
    /// Returns the net fluid-on-obstacle force `(fx, fy)` accumulated during the last stream step.
    #[verified_engine::verified]
    pub fn net_obstacle_force(&self) -> (f64, f64) {
        (self.last_drag_force, self.last_lift_force)
    }

    /// Counts the number of solid obstacle cells currently active on the lattice.
    #[verified_engine::verified]
    pub fn obstacle_cell_count(&self) -> usize {
        let mut count = 0usize;
        for &is_obs in self.obstacles.as_slice() {
            if is_obs {
                count += 1;
            }
        }
        count
    }
}

impl<const Q: usize, L: Lattice2D<Q>, C: CollisionModel<Q, L>> LatticeBoltzmann<Q, L, C> {
    /// Returns the instantaneous hydrodynamic force `(fx, fy)` exerted by the fluid on all
    /// solid obstacles via momentum-exchange bounce-back reflections.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_physics::__doctest_facade as math_explorer;
    /// use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
    ///
    /// let mut solver = LatticeBoltzmannD2Q9::new(20, 10, 0.7);
    /// solver.set_obstacle(10, 5, true);
    /// for _ in 0..15 {
    ///     solver.set_inlet(0, 0, 2, 10, 0.1, 0.0);
    ///     solver.step();
    /// }
    /// let (fx, _fy) = solver.net_obstacle_force();
    /// assert!(fx > 0.0);
    /// ```
    #[verified_engine::verified]
    pub fn net_obstacle_force(&self) -> (f64, f64) {
        self.state.net_obstacle_force()
    }

    /// Computes the scalar 2D vorticity curl $\omega_z = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}$
    /// at `(x, y)` using second-order central differences on interior nodes and first-order
    /// differences on domain boundaries. Returns `0.0` inside solid obstacles or out-of-bounds coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_physics::__doctest_facade as math_explorer;
    /// use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
    ///
    /// let solver = LatticeBoltzmannD2Q9::new(10, 10, 0.8);
    /// assert_eq!(solver.vorticity_curl(5, 5), 0.0);
    /// ```
    #[verified_engine::verified]
    pub fn vorticity_curl(&self, x: usize, y: usize) -> f64 {
        let width = self.state.width;
        let height = self.state.height;
        if x >= width || y >= height || width < 2 || height < 2 || self.state.obstacles[(y, x)] {
            return 0.0;
        }
        let x_prev = x.saturating_sub(1);
        let x_next = (x + 1).min(width - 1);
        let y_prev = y.saturating_sub(1);
        let y_next = (y + 1).min(height - 1);

        let dx = (x_next - x_prev) as f64;
        let dy = (y_next - y_prev) as f64;
        let duy_dx = (self.state.uy[(y, x_next)] - self.state.uy[(y, x_prev)]) / dx;
        let dux_dy = (self.state.ux[(y_next, x)] - self.state.ux[(y_prev, x)]) / dy;
        duy_dx - dux_dy
    }

    /// Returns the total number of cells currently marked as solid obstacles.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_physics::__doctest_facade as math_explorer;
    /// use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
    ///
    /// let mut solver = LatticeBoltzmannD2Q9::new(10, 10, 1.0);
    /// assert_eq!(solver.obstacle_cell_count(), 0);
    /// solver.set_obstacle(4, 4, true);
    /// assert_eq!(solver.obstacle_cell_count(), 1);
    /// ```
    #[verified_engine::verified]
    pub fn obstacle_cell_count(&self) -> usize {
        self.state.obstacle_cell_count()
    }
}
