use super::{
    CoupledVortexMorphogenesis, DimensionlessRegimeMetrics, MAX_ADVECTIVE_CFL, MAX_DIFFUSIVE_CFL,
    MAX_TRANSPORT_SUBSTEPS, MorphogenKinetics, VELOCITY_SAFETY_CAP,
};
use math_commons::math_kernel::types::flatten_2d_index;

impl CoupledVortexMorphogenesis {
    #[verified_engine::verified]
    pub(super) fn required_substeps(&self) -> usize {
        let dt = self.config.dt.max(0.0);
        if dt == 0.0 {
            return 1;
        }
        let mut max_l1_u = 0.0_f64;
        for i in 0..self.ux_row.len() {
            let l1 = self.ux_row[i].abs() + self.uy_row[i].abs();
            if l1 > max_l1_u {
                max_l1_u = l1;
            }
        }
        let dx = self.config.dx.max(0.05);
        let inv_dx = 1.0 / dx;
        let d_max = self.config.d_u.max(self.config.d_v).max(0.0);
        let adv_need = (max_l1_u * inv_dx * dt) / MAX_ADVECTIVE_CFL;
        let diff_need = (4.0 * d_max * inv_dx * inv_dx * dt) / MAX_DIFFUSIVE_CFL;
        let needed = adv_need.max(diff_need).ceil() as usize;
        needed.clamp(1, MAX_TRANSPORT_SUBSTEPS)
    }

    #[verified_engine::verified]
    pub(super) fn apply_inlet_drive(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let u_in = self
            .config
            .inlet_ux
            .clamp(-VELOCITY_SAFETY_CAP, VELOCITY_SAFETY_CAP);
        if self.config.shear_profile {
            let denom = (h.saturating_sub(1)).max(1) as f64;
            for y in 1..h - 1 {
                let s = (y as f64) / denom;
                let u_shear = u_in * (2.0 * s - 1.0);
                self.lbm.set_inlet(0, y, w, 1, u_shear, 0.0);
            }
        } else {
            self.lbm.set_inlet(0, 1, 2, h.saturating_sub(2), u_in, 0.0);
        }
    }

    #[verified_engine::verified]
    pub(super) fn apply_morphogen_sources_and_inlet(&mut self, dt_sub: f64) {
        let w = self.config.width;
        let h = self.config.height;
        let (u_feed, v_feed) = self.config.kinetics.homogeneous_feed();
        let rate = (self.config.emitter_strength.clamp(0.0, 1.0) * dt_sub).clamp(0.0, 1.0);
        let kinetics = self.config.kinetics;
        let (u_slice, v_slice) = self.morphogens.u_and_v_mut();

        if self.config.fresh_inlet_feed {
            for y in 0..h {
                let idx = flatten_2d_index(0, y, w);
                if !self.obstacles_row[idx] && self.ux_row[idx] > 0.0 {
                    u_slice[idx] = u_feed;
                    v_slice[idx] = v_feed;
                }
            }
        }

        if rate > 0.0 {
            for i in 0..u_slice.len() {
                if self.emitters_row[i] && !self.obstacles_row[i] {
                    match kinetics {
                        MorphogenKinetics::GrayScott(_) => {
                            u_slice[i] += rate * (0.48 - u_slice[i]);
                            v_slice[i] += rate * (0.28 - v_slice[i]);
                        }
                        MorphogenKinetics::Schnakenberg(sn) => {
                            let target_u = (sn.a + sn.b) * 1.65;
                            u_slice[i] += rate * (target_u - u_slice[i]);
                        }
                    }
                }
            }
        }
    }

    #[verified_engine::verified]
    pub(super) fn compute_vorticity(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        if w < 3 || h < 3 {
            self.vorticity_row.fill(0.0);
            return;
        }
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                if self.obstacles_row[idx] || x == 0 || x + 1 >= w || y == 0 || y + 1 >= h {
                    self.vorticity_row[idx] = 0.0;
                    continue;
                }
                let dv_dx = 0.5 * (self.uy_row[idx + 1] - self.uy_row[idx - 1]);
                let du_dy = 0.5 * (self.ux_row[idx + w] - self.ux_row[idx - w]);
                self.vorticity_row[idx] = dv_dx - du_dy;
            }
        }
    }

    #[verified_engine::verified]
    pub(super) fn update_metrics(&mut self, max_speed: f64, obs_count: usize, substeps: usize) {
        let dx = self.config.dx.max(0.05);
        let l_cells = if obs_count > 0 {
            (2.0 * ((obs_count as f64) / std::f64::consts::PI).sqrt()).max(4.0)
        } else {
            (self.config.height as f64 * 0.5).max(4.0)
        };
        let l_char = l_cells * dx;
        let u_ref = max_speed.max(
            (self.config.inlet_ux.abs() * self.config.advection_coupling).min(VELOCITY_SAFETY_CAP),
        );
        let d_slow = self.config.d_u.min(self.config.d_v).max(1e-6);
        let peclet = (u_ref * l_char) / d_slow;
        let k_react = self.config.kinetics.characteristic_rate();
        let damkohler = (k_react * l_char) / u_ref.max(1e-4);
        self.last_metrics = DimensionlessRegimeMetrics {
            peclet,
            damkohler,
            max_speed,
            substeps,
        };
    }

    #[verified_engine::verified]
    pub(super) fn seed_catalyst_cell(&mut self, idx: usize) {
        match self.config.kinetics {
            MorphogenKinetics::GrayScott(_) => {
                self.morphogens.u_mut()[idx] = 0.50;
                self.morphogens.v_mut()[idx] = 0.28;
            }
            MorphogenKinetics::Schnakenberg(sn) => {
                let u0 = (sn.a + sn.b).max(0.1);
                self.morphogens.u_mut()[idx] = u0 * 1.4;
                self.morphogens.v_mut()[idx] = sn.b / (u0 * u0);
            }
        }
    }

    #[verified_engine::verified]
    pub(super) fn seed_karman_spots(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let cx = w / 5;
        let cy = h / 2;
        self.paint_obstacle_disc(cx, cy, 4, true);
        self.lbm.set_inlet(0, 1, w, h.saturating_sub(2), 0.10, 0.0);
        for y in 2..h.saturating_sub(2) {
            let dy = (y as f64) - (cy as f64);
            let env_y = (-(dy * dy) / 36.0).exp();
            let def_y = (-(dy * dy) / 18.0).exp();
            for x in (cx + 5)..w.saturating_sub(1) {
                let s = (x - cx) as f64;
                let ux = 0.10 * (1.0 - 0.55 * (-s / 32.0).exp() * def_y);
                let phase = (2.0 * std::f64::consts::PI * s) / 24.0;
                let uy = 0.036 * phase.sin() * env_y - 0.014 * (dy / 6.0) * phase.cos() * env_y;
                self.lbm.set_inlet(x, y, 1, 1, ux, uy);
            }
        }
        for _ in 0..12 {
            self.lbm.set_inlet(0, 1, 2, h.saturating_sub(2), 0.10, 0.0);
            self.lbm.step();
            self.lbm.apply_open_outlet();
        }
        self.paint_emitter_disc(cx + 7, cy, 2, true);
        self.paint_emitter_disc(cx + 16, cy + 2, 2, true);
    }

    #[verified_engine::verified]
    pub(super) fn seed_shear_stripes(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let (u0, v0) = self.config.kinetics.homogeneous_feed();
        let (u_slice, v_slice) = self.morphogens.u_and_v_mut();
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                let px = ((x * 7 + y * 13) % 19) as f64 / 19.0 - 0.5;
                let py = ((x * 11 + y * 5) % 17) as f64 / 17.0 - 0.5;
                u_slice[idx] = (u0 + 0.12 * px).max(0.01);
                v_slice[idx] = (v0 + 0.08 * py).max(0.01);
            }
        }
        self.apply_inlet_drive();
    }

    #[verified_engine::verified]
    pub(super) fn seed_stagnation_plumes(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let cx = (w * 2) / 5;
        let cy = h / 2;
        self.paint_obstacle_disc(cx, cy, 6, true);
        self.lbm.set_inlet(0, 1, w, h.saturating_sub(2), 0.09, 0.0);
        for y in 2..h.saturating_sub(2) {
            let dy = (y as f64) - (cy as f64);
            let wake = (-(dy * dy) / 28.0).exp();
            for x in (cx + 6)..w.saturating_sub(1) {
                let s = (x - cx) as f64;
                let ux = 0.09 * (1.0 - 0.88 * (-s / 26.0).exp() * wake);
                self.lbm.set_inlet(x, y, 1, 1, ux, 0.0);
            }
        }
        for _ in 0..12 {
            self.lbm.set_inlet(0, 1, 2, h.saturating_sub(2), 0.09, 0.0);
            self.lbm.step();
            self.lbm.apply_open_outlet();
        }
        self.paint_emitter_disc(cx + 8, cy, 2, true);
    }

    #[verified_engine::verified]
    pub(super) fn seed_damkohler_washout(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        // Small wall-mounted microfluidic ribs keep central channel core unobstructed
        self.paint_obstacle_disc(w / 6, 3, 1, true);
        self.paint_obstacle_disc(w / 6, h.saturating_sub(4), 1, true);
        self.lbm.set_inlet(0, 1, w, h.saturating_sub(2), 0.145, 0.0);
        // Seed transient autocatalytic spots in the upstream central core without continuous emitters
        for cx in [w / 8, w / 5] {
            for y in (h / 2 - 4)..=(h / 2 + 4) {
                for x in (cx - 3)..=(cx + 3) {
                    let idx = flatten_2d_index(x, y, w);
                    if !self.obstacles_row[idx] {
                        self.seed_catalyst_cell(idx);
                    }
                }
            }
        }
    }
}
