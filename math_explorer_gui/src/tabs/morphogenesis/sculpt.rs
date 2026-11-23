#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MorphogenesisRunner;
#[cfg(feature = "physics")]
use super::types::MorphogenesisFlowMode;
use super::types::{MorphogenesisBrushMode, MorphogenesisKineticsMode};
use math_commons::math_kernel::types::flatten_2d_index;
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, UniformFlow, Vortex,
};

impl MorphogenesisRunner {
    pub(crate) fn seed_default_bluff_obstacle(&mut self) {
        if self.width < 8 || self.height < 8 {
            return;
        }
        let cx = self.width / 4;
        let cy = self.height / 2;
        let radius = (self.height / 12).clamp(2, 6);
        self.paint_obstacle_both(cx, cy, radius, true);
    }

    pub(crate) fn paint_obstacle_both(
        &mut self,
        cx: usize,
        cy: usize,
        radius: usize,
        is_obstacle: bool,
    ) {
        self.system
            .diffusion
            .paint_obstacle_disk(cx, cy, radius, is_obstacle);
        #[cfg(feature = "physics")]
        {
            let r = radius as isize;
            let y_min = cy.saturating_sub(radius);
            let y_max = (cy + radius).min(self.height.saturating_sub(1));
            let x_min = cx.saturating_sub(radius);
            let x_max = (cx + radius).min(self.width.saturating_sub(1));
            for y in y_min..=y_max {
                for x in x_min..=x_max {
                    let dx = x as isize - cx as isize;
                    let dy = y as isize - cy as isize;
                    if dx * dx + dy * dy <= r * r {
                        self.lbm.set_obstacle(x, y, is_obstacle);
                    }
                }
            }
            self.potential_dirty = true;
        }
    }

    pub(crate) fn apply_reagent_disk(&mut self, cx: usize, cy: usize, radius: usize, inject: bool) {
        let r = radius as isize;
        let y_min = cy.saturating_sub(radius);
        let y_max = (cy + radius).min(self.height.saturating_sub(1));
        let x_min = cx.saturating_sub(radius);
        let x_max = (cx + radius).min(self.width.saturating_sub(1));
        let mode = self.system.kinetics.mode;
        let a = self.system.kinetics.a;
        let b = self.system.kinetics.b;

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x as isize - cx as isize;
                let dy = y as isize - cy as isize;
                let idx = flatten_2d_index(x, y, self.width);
                if dx * dx + dy * dy <= r * r && !self.system.diffusion.obstacles()[idx] {
                    let (u_val, v_val) = match (mode, inject) {
                        (MorphogenesisKineticsMode::Schnakenberg, true) => {
                            let u_eq = (a + b).max(0.05);
                            (u_eq + 1.2, b / (u_eq * u_eq))
                        }
                        (MorphogenesisKineticsMode::Schnakenberg, false) => {
                            let u_eq = (a + b).max(0.05);
                            (u_eq, b / (u_eq * u_eq))
                        }
                        (MorphogenesisKineticsMode::GrayScott, true) => (0.50, 0.25),
                        (MorphogenesisKineticsMode::GrayScott, false) => (1.0, 0.0),
                    };
                    self.system.u_mut()[idx] = u_val;
                    self.system.v_mut()[idx] = v_val;
                }
            }
        }
    }

    #[cfg(feature = "physics")]
    pub(crate) fn rebuild_potential_elements(&mut self, inlet_velocity: f64, circulation: f64) {
        let key = (inlet_velocity.to_bits(), circulation.to_bits());
        if self.last_potential_key == key && !self.potential_dirty {
            return;
        }
        self.potential_flow.clear();
        let cx = (self.width as f64) * 0.35;
        let cy = (self.height as f64) * 0.50;
        let cyl_r = ((self.height as f64) / 10.0).max(2.0);
        let kappa = 2.0 * std::f64::consts::PI * inlet_velocity * cyl_r * cyl_r;
        self.potential_flow
            .add(Box::new(UniformFlow::new(inlet_velocity, 0.0)));
        self.potential_flow
            .add(Box::new(Doublet::new(kappa, cx, cy)));
        self.potential_flow
            .add(Box::new(Vortex::new(circulation, cx, cy)));
        self.last_potential_key = key;
        self.potential_dirty = false;
        self.evaluate_potential_grid();
    }

    #[cfg(feature = "physics")]
    fn evaluate_potential_grid(&mut self) {
        let w = self.width;
        let h = self.height;
        let (vx, vy, obs) = self.system.diffusion.buffers_mut();
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                if obs[idx] {
                    vx[idx] = 0.0;
                    vy[idx] = 0.0;
                } else {
                    let vel = self.potential_flow.velocity(x as f64 + 0.5, y as f64 + 0.5);
                    vx[idx] = vel.x.clamp(-1.5, 1.5);
                    vy[idx] = vel.y.clamp(-1.5, 1.5);
                }
            }
        }
    }

    #[cfg(feature = "physics")]
    pub(crate) fn step_lbm_coupling(&mut self, inlet_velocity: f64, viscosity: f64) {
        self.lbm.collision_model.tau = (3.0 * viscosity + 0.5).max(0.51);
        let u_in = inlet_velocity.clamp(0.0, 0.22);
        let jet_h = self.height.saturating_sub(2).max(1);
        let jet_y = usize::from(self.height > 2);
        self.lbm.set_inlet(0, jet_y, 1, jet_h, u_in, 0.0);
        self.lbm.step();
        let (vx, vy, obs) = self.system.diffusion.buffers_mut();
        self.lbm.state.copy_macroscopic_row_major(vx, vy, obs);
    }

    pub(crate) fn handle_brush_command(&mut self, cx: i32, cy: i32, r: i32, is_obstacle: bool) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let gx = cx.clamp(0, self.width as i32 - 1) as usize;
        let gy = cy.clamp(0, self.height as i32 - 1) as usize;
        let radius = r.max(0) as usize;
        let snap = self.read_params();
        match snap.brush_mode {
            MorphogenesisBrushMode::Obstacle => {
                self.paint_obstacle_both(gx, gy, radius, is_obstacle);
            }
            MorphogenesisBrushMode::EraseObstacle => {
                self.paint_obstacle_both(gx, gy, radius, false);
            }
            MorphogenesisBrushMode::Reagent => {
                self.apply_reagent_disk(gx, gy, radius, is_obstacle);
            }
            MorphogenesisBrushMode::PanZoom => {}
        }
        #[cfg(feature = "physics")]
        if snap.flow_mode == MorphogenesisFlowMode::PotentialFlow && self.potential_dirty {
            self.rebuild_potential_elements(snap.inlet_velocity, snap.circulation);
        }
        self.refresh_diagnostics(&snap);
    }

    pub(crate) fn handle_clear_obstacles(&mut self) {
        self.system.diffusion.clear_obstacles();
        #[cfg(feature = "physics")]
        {
            self.lbm.clear_obstacles();
            self.potential_dirty = true;
        }
        let snap = self.read_params();
        #[cfg(feature = "physics")]
        if snap.flow_mode == MorphogenesisFlowMode::PotentialFlow {
            self.rebuild_potential_elements(snap.inlet_velocity, snap.circulation);
        }
        self.refresh_diagnostics(&snap);
    }
}
