//! Hydrodynamic LBM / PotentialFlow velocity synchronization and obstacle/reagent sculpting
//! helpers for [`super::runner::MorphogenesisKineticsRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MorphogenesisKineticsRunner;
use super::types::{
    MorphogenesisBrushMode, MorphogenesisFlowMode, MorphogenesisFlowState,
    MorphogenesisKineticsMode,
};
use math_commons::math_kernel::types::flatten_2d_index;
#[cfg(feature = "physics")]
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, UniformFlow, Vortex,
};

impl MorphogenesisKineticsRunner {
    pub(crate) fn sync_mask_to_hydrodynamics(&mut self) {
        self.system.diffusion.sync_obstacles_from_mask();
        #[cfg(feature = "physics")]
        {
            for y in 0..self.height {
                for x in 0..self.width {
                    let is_obs =
                        self.system.diffusion.obstacles()[flatten_2d_index(x, y, self.width)];
                    self.lbm.set_obstacle(x, y, is_obs);
                }
            }
            self.potential_dirty = true;
        }
        let flow = self.read_flow();
        #[cfg(feature = "physics")]
        if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
            self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
        }
        if flow.flow_mode != MorphogenesisFlowMode::None {
            self.refresh_diagnostics();
        }
    }

    pub(crate) fn seed_default_bluff_obstacle(&mut self) {
        if self.width < 8 || self.height < 8 {
            return;
        }
        let cx = self.width / 4;
        let cy = self.height / 2;
        let radius = (self.height / 12).clamp(2, 6);
        self.system
            .diffusion
            .advection
            .paint_obstacle_disk(cx, cy, radius, true);
        #[cfg(feature = "physics")]
        {
            for y in 0..self.height {
                for x in 0..self.width {
                    let is_obs =
                        self.system.diffusion.obstacles()[flatten_2d_index(x, y, self.width)];
                    self.lbm.set_obstacle(x, y, is_obs);
                }
            }
            self.potential_dirty = true;
            let flow = self.read_flow();
            if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
                self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
            }
        }
        self.refresh_diagnostics();
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
        self.enforce_mask_concentrations();
        #[cfg(feature = "physics")]
        {
            for y in 0..self.height {
                for x in 0..self.width {
                    let is_obs =
                        self.system.diffusion.obstacles()[flatten_2d_index(x, y, self.width)];
                    self.lbm.set_obstacle(x, y, is_obs);
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

    pub(crate) fn sync_flow_field(&mut self, flow: &MorphogenesisFlowState) {
        let active = flow.flow_mode != MorphogenesisFlowMode::None;
        self.system.diffusion.flow_active = active;
        if self.last_flow_mode != flow.flow_mode {
            if !active {
                self.system.diffusion.advection.clear_velocity();
                self.peclet = 0.0;
            } else if !self.system.diffusion.obstacles().iter().any(|&b| b) {
                self.seed_default_bluff_obstacle();
            }
            self.last_flow_mode = flow.flow_mode;
            #[cfg(feature = "physics")]
            {
                self.potential_dirty = true;
            }
        }
        match flow.flow_mode {
            MorphogenesisFlowMode::None => {}
            MorphogenesisFlowMode::LbmVortexWake => {
                #[cfg(feature = "physics")]
                self.step_lbm_coupling(flow.inlet_velocity, flow.viscosity);
                #[cfg(not(feature = "physics"))]
                self.system
                    .diffusion
                    .advection
                    .set_uniform_velocity(flow.inlet_velocity, 0.0);
            }
            MorphogenesisFlowMode::PotentialFlow => {
                #[cfg(feature = "physics")]
                self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
                #[cfg(not(feature = "physics"))]
                self.system
                    .diffusion
                    .advection
                    .set_uniform_velocity(flow.inlet_velocity, 0.0);
            }
        }
        if let Some(ref mut twin) = self.twin {
            twin.system_a.diffusion.flow_active = active;
            if active {
                let (src_vx, src_vy, _) = self.system.diffusion.advection.buffers_mut();
                let (dst_vx, dst_vy, _) = twin.system_a.diffusion.advection.buffers_mut();
                dst_vx.copy_from_slice(src_vx);
                dst_vy.copy_from_slice(src_vy);
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
        let (vx, vy, obs) = self.system.diffusion.advection.buffers_mut();
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
        let (vx, vy, obs) = self.system.diffusion.advection.buffers_mut();
        self.lbm.state.copy_macroscopic_row_major(vx, vy, obs);
    }

    pub(crate) fn handle_brush_command(&mut self, cx: i32, cy: i32, r: i32, is_obstacle: bool) {
        if self.width == 0 || self.height == 0 {
            return;
        }
        let gx = cx.clamp(0, self.width as i32 - 1) as usize;
        let gy = cy.clamp(0, self.height as i32 - 1) as usize;
        let radius = r.max(0) as usize;
        let flow = self.read_flow();
        match flow.brush_mode {
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
        if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow && self.potential_dirty {
            self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
        }
        self.refresh_diagnostics();
    }

    pub(crate) fn handle_clear_obstacles(&mut self) {
        self.system.diffusion.clear_all_obstacles();
        #[cfg(feature = "physics")]
        {
            self.lbm.clear_obstacles();
            self.potential_dirty = true;
            let flow = self.read_flow();
            if flow.flow_mode == MorphogenesisFlowMode::PotentialFlow {
                self.rebuild_potential_elements(flow.inlet_velocity, flow.circulation);
            }
        }
        self.refresh_diagnostics();
    }
}
