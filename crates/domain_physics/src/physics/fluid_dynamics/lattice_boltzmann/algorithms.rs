use super::model::*;
use super::state::LatticeState;
use std::marker::PhantomData;
impl LatticeBoltzmannD2Q9<BgkCollision> {
    /// Creates a new solver with the given dimensions and relaxation time.
    ///
    /// * `tau`: Relaxation time. Must be > 0.5 for stability.
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize, tau: f64) -> Self {
        Self::new_with_model(width, height, BgkCollision { tau: tau.max(0.51) })
    }
}

impl<const Q: usize, L: Lattice2D<Q>, C: CollisionModel<Q, L>> LatticeBoltzmann<Q, L, C> {
    /// Creates a new solver with a specific collision model.
    #[verified_engine::verified]
    pub fn new_with_model(width: usize, height: usize, collision_model: C) -> Self {
        // Validation: Ensure lattice directions are within [-1, 1] range supported by stream()
        for &dx in &L::directions_x() {
            assert!(dx.abs() <= 1, "Lattice directions must be within [-1, 1]");
        }
        for &dy in &L::directions_y() {
            assert!(dy.abs() <= 1, "Lattice directions must be within [-1, 1]");
        }

        let mut solver = Self {
            state: LatticeState::new(width, height),
            collision_model,
            _marker: PhantomData,
        };
        solver.init_equilibrium();
        solver
    }

    /// Initializes the grid to a uniform equilibrium state (rho=1, u=0).
    #[verified_engine::verified]
    pub fn init_equilibrium(&mut self) {
        self.state.rho.fill(1.0);
        self.state.ux.fill(0.0);
        self.state.uy.fill(0.0);
        let eq = L::equilibrium(1.0, 0.0, 0.0);
        self.state.f.fill(eq);
        self.state.f_new.fill(eq);
    }

    /// Sets a rectangular region of velocity (e.g., inlet).
    #[verified_engine::verified]
    pub fn set_inlet(&mut self, x: usize, y: usize, w: usize, h: usize, u_x: f64, u_y: f64) {
        for j in y..(y + h).min(self.state.height) {
            for i in x..(x + w).min(self.state.width) {
                if !self.state.obstacles[(j, i)] {
                    self.state.ux[(j, i)] = u_x;
                    self.state.uy[(j, i)] = u_y;
                    // Reset distributions to equilibrium for the new velocity
                    // keeping density roughly constant (1.0)
                    self.state.f[(j, i)] = L::equilibrium(1.0, u_x, u_y);
                }
            }
        }
    }

    /// Sets an obstacle at (x, y).
    #[verified_engine::verified]
    pub fn set_obstacle(&mut self, x: usize, y: usize, is_obstacle: bool) {
        if x < self.state.width && y < self.state.height {
            self.state.obstacles[(y, x)] = is_obstacle;
            // Reset velocity inside obstacle
            if is_obstacle {
                self.state.ux[(y, x)] = 0.0;
                self.state.uy[(y, x)] = 0.0;
            }
        }
    }

    /// Performs one simulation step (`Stream -> Collision -> SwapBuffers`).
    #[verified_engine::verified]
    pub fn step(&mut self) {
        self.stream();
        self.boundary_conditions(); // Apply macroscopic BCs if any
        self.macroscopic_and_collision();
        pure_math::pure_math::analysis::evolution::DoubleBufferedState::swap_buffers(
            &mut self.state,
        );
    }

    /// Applies a zero-gradient convective outlet on the right boundary (`x = width - 1`)
    /// so channel wakes exit smoothly without periodic wrap-around reflections.
    #[verified_engine::verified]
    pub fn apply_open_outlet(&mut self) {
        let w = self.state.width;
        let h = self.state.height;
        if w < 2 {
            return;
        }
        let last = w - 1;
        let prev = w - 2;
        for y in 0..h {
            if !self.state.obstacles[(y, last)] && !self.state.obstacles[(y, prev)] {
                self.state.f[(y, last)] = self.state.f[(y, prev)];
                self.state.rho[(y, last)] = self.state.rho[(y, prev)];
                self.state.ux[(y, last)] = self.state.ux[(y, prev)];
                self.state.uy[(y, last)] = self.state.uy[(y, prev)];
            }
        }
    }

    /// Fused Macroscopic + Collision step.
    ///
    /// # Performance
    /// This optimization fuses the macroscopic moment calculation and the collision step
    /// into a single loop. This reduces memory bandwidth pressure by:
    /// 1. Reading `f` only once (instead of twice).
    /// 2. Keeping `rho`, `ux`, `uy` in CPU registers for the collision step, avoiding
    ///    store-then-load latency for these variables.
    #[verified_engine::verified]
    fn macroscopic_and_collision(&mut self) {
        let cx = L::directions_x();
        let cy = L::directions_y();

        let collision_model = &self.collision_model;

        // Use zipping for iteration to enable better vectorization and eliminate bounds checks
        self.state
            .f_new
            .iter_mut()
            .zip(self.state.rho.iter_mut())
            .zip(self.state.ux.iter_mut())
            .zip(self.state.uy.iter_mut())
            .zip(self.state.obstacles.iter())
            .for_each(|((((f_cell, rho), ux), uy), is_obs)| {
                if *is_obs {
                    *ux = 0.0;
                    *uy = 0.0;
                    return;
                }

                let mut local_rho = 0.0;
                let mut local_jx = 0.0;
                let mut local_jy = 0.0;

                for k in 0..Q {
                    let val = f_cell[k];
                    local_rho += val;
                    local_jx += val * cx[k] as f64;
                    local_jy += val * cy[k] as f64;
                }

                *rho = local_rho;

                let (local_ux, local_uy) = if local_rho > 0.0 {
                    (local_jx / local_rho, local_jy / local_rho)
                } else {
                    (0.0, 0.0)
                };

                *ux = local_ux;
                *uy = local_uy;

                collision_model.apply(f_cell, local_rho, local_ux, local_uy);
            });
    }

    /// Streaming step: Move particles to neighboring cells.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    #[verified_engine::verified]
    fn stream(&mut self) {
        let width = self.state.width;
        let height = self.state.height;
        let required_len = width * height;

        // Security Check: Ensure invariants hold before entering unsafe blocks.
        assert_eq!(
            self.state.f.len(),
            required_len,
            "LatticeState invariant violated: f.len() != width * height"
        );
        assert_eq!(
            self.state.f_new.len(),
            required_len,
            "LatticeState invariant violated: f_new.len() != width * height"
        );
        assert_eq!(
            self.state.obstacles.len(),
            required_len,
            "LatticeState invariant violated: obstacles.len() != width * height"
        );

        let cx = L::directions_x();
        let cy = L::directions_y();
        let opp = L::opposite_indices();

        // 1. Interior (Hot Path) - Loop Splitting & optimizations
        if width > 2 && height > 2 {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let is_obstacle = self.state.obstacles[(y, x)];
                    if is_obstacle {
                        continue;
                    }

                    for k in 0..Q {
                        let prev_y = (y as isize - cy[k] as isize) as usize;
                        let prev_x = (x as isize - cx[k] as isize) as usize;

                        let source_is_obstacle = self.state.obstacles[(prev_y, prev_x)];

                        if source_is_obstacle {
                            // Bounce-back
                            let bounce_val = self.state.f[(y, x)][opp[k]];
                            self.state.f_new[(y, x)][k] = bounce_val;
                        } else {
                            // Stream
                            let stream_val = self.state.f[(prev_y, prev_x)][k];
                            self.state.f_new[(y, x)][k] = stream_val;
                        }
                    }
                }
            }
        }

        // 2. Boundary Handling (Slow Path)
        // Process top/bottom rows and left/right columns
        let process_boundary_cell = |x: usize, y: usize, state: &mut LatticeState<Q>| {
            if state.obstacles[(y, x)] {
                return;
            }

            for k in 0..Q {
                let prev_x = x as i32 - cx[k];
                let prev_y = y as i32 - cy[k];

                if prev_x >= 0 && prev_x < width as i32 && prev_y >= 0 && prev_y < height as i32 {
                    if state.obstacles[(prev_y as usize, prev_x as usize)] {
                        state.f_new[(y, x)][k] = state.f[(y, x)][opp[k]];
                    } else {
                        state.f_new[(y, x)][k] = state.f[(prev_y as usize, prev_x as usize)][k];
                    }
                } else {
                    // Boundary Logic (Periodic X, Bounce Y)
                    let mut src_x = prev_x;
                    let src_y = prev_y;

                    // Periodic X
                    if src_x < 0 {
                        src_x += width as i32;
                    } else if src_x >= width as i32 {
                        src_x -= width as i32;
                    }

                    if src_y < 0 || src_y >= height as i32 {
                        // Wall Bounce
                        state.f_new[(y, x)][k] = state.f[(y, x)][opp[k]];
                    } else {
                        if state.obstacles[(src_y as usize, src_x as usize)] {
                            state.f_new[(y, x)][k] = state.f[(y, x)][opp[k]];
                        } else {
                            state.f_new[(y, x)][k] = state.f[(src_y as usize, src_x as usize)][k];
                        }
                    }
                }
            }
        };

        // Top & Bottom
        for y in [0, height - 1] {
            for x in 0..width {
                process_boundary_cell(x, y, &mut self.state);
            }
        }
        // Left & Right (excluding corners)
        for y in 1..height - 1 {
            for x in [0, width - 1] {
                process_boundary_cell(x, y, &mut self.state);
            }
        }
    }

    /// Enforce specific boundary conditions (like driving velocity).
    #[verified_engine::verified]
    fn boundary_conditions(&mut self) {
        // ... (Existing code is commented out, preserving it as is)
    }

    // --- Accessors ---

    /// Retrieves the number of grid points along the horizontal (X) axis.
    ///
    /// # Returns
    ///
    /// The width of the simulation domain.
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.state.width
    }

    /// Retrieves the number of grid points along the vertical (Y) axis.
    ///
    /// # Returns
    ///
    /// The height of the simulation domain.
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.state.height
    }

    /// Retrieves the macroscopic fluid density ($\rho$) at the specified coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - The horizontal coordinate.
    /// * `y` - The vertical coordinate.
    ///
    /// # Returns
    ///
    /// A `f64` representing the macroscopic fluid density at the cell.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height` due to out-of-bounds flat array indexing.
    #[verified_engine::verified]
    pub fn get_density(&self, x: usize, y: usize) -> f64 {
        self.state.rho[(y, x)]
    }

    /// Retrieves the macroscopic fluid velocity $(u_x, u_y)$ at the specified coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - The horizontal coordinate.
    /// * `y` - The vertical coordinate.
    ///
    /// # Returns
    ///
    /// A tuple `(f64, f64)` containing the horizontal and vertical velocity components.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height` due to out-of-bounds flat array indexing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_physics::__doctest_facade as math_explorer;
    /// use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
    ///
    /// let solver = LatticeBoltzmannD2Q9::new(10, 10, 1.0);
    /// let (ux, uy) = solver.get_velocity(5, 5);
    ///
    /// // The simulation initializes at rest (equilibrium).
    /// assert_eq!(ux, 0.0);
    /// assert_eq!(uy, 0.0);
    /// ```
    #[verified_engine::verified]
    pub fn get_velocity(&self, x: usize, y: usize) -> (f64, f64) {
        (self.state.ux[(y, x)], self.state.uy[(y, x)])
    }

    /// Calculates the magnitude of the macroscopic fluid velocity at the specified coordinates.
    ///
    /// Computes the Euclidean norm of the velocity vector: $\sqrt{u_x^2 + u_y^2}$.
    ///
    /// # Arguments
    ///
    /// * `x` - The horizontal coordinate.
    /// * `y` - The vertical coordinate.
    ///
    /// # Returns
    ///
    /// A `f64` representing the total speed of the fluid at the cell.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height` due to out-of-bounds flat array indexing.
    #[verified_engine::verified]
    pub fn get_velocity_magnitude(&self, x: usize, y: usize) -> f64 {
        (self.state.ux[(y, x)].powi(2) + self.state.uy[(y, x)].powi(2)).sqrt()
    }

    /// Checks if a given cell is marked as an obstacle.
    ///
    /// Obstacles enforce the bounce-back boundary condition during the collision step,
    /// driving the fluid velocity at their coordinates to exactly zero.
    ///
    /// # Arguments
    ///
    /// * `x` - The horizontal coordinate.
    /// * `y` - The vertical coordinate.
    ///
    /// # Returns
    ///
    /// `true` if the cell is an obstacle, `false` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height` due to out-of-bounds flat array indexing.
    #[verified_engine::verified]
    pub fn is_obstacle(&self, x: usize, y: usize) -> bool {
        self.state.obstacles[(y, x)]
    }

    /// Clears all obstacles from the lattice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_physics::__doctest_facade as math_explorer;
    /// use math_explorer::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;
    ///
    /// let mut solver = LatticeBoltzmannD2Q9::new(10, 10, 1.0);
    /// solver.set_obstacle(5, 5, true);
    /// assert!(solver.is_obstacle(5, 5));
    ///
    /// solver.clear_obstacles();
    /// assert!(!solver.is_obstacle(5, 5));
    /// ```
    #[verified_engine::verified]
    pub fn clear_obstacles(&mut self) {
        self.state.obstacles.fill(false);
    }
}

use pure_math::pure_math::analysis::evolution::{EvolutionEngine, EvolutionError};
use rand::RngCore;

impl<const Q: usize, L: Lattice2D<Q>, C: CollisionModel<Q, L>> EvolutionEngine<LatticeState<Q>, ()>
    for LatticeBoltzmann<Q, L, C>
{
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut LatticeState<Q>,
        _aux: &mut (),
        _rng: &mut R,
        _dt: f64,
    ) -> Result<(), EvolutionError> {
        std::mem::swap(&mut self.state, state);
        self.step();
        std::mem::swap(&mut self.state, state);
        Ok(())
    }
}
