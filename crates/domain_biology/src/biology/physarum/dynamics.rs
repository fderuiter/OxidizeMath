//! Internal zero-allocation Lagrangian-Eulerian stepping physics for
//! [`crate::biology::physarum::PhysarumEngine`].

use super::engine::PhysarumEngine;
use super::{MAX_PHYSARUM_AGENTS, MAX_PHYSARUM_TERMINALS, MIN_PHYSARUM_AGENTS};
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use std::f64::consts::{PI, TAU};

pub(crate) const MAX_TRAIL_CONCENTRATION: f64 = 120.0;
const PRUNE_HALF_SATURATION: f64 = 0.90;
pub(crate) const ACTIVE_TUBE_THRESHOLD: f64 = 0.30;
const ENERGY_DECAY_PER_STEP: f64 = 0.980;
const MIN_AGENT_ENERGY: f64 = 0.08;
const INV_U64_MAX: f64 = 1.0 / (u64::MAX as f64);

#[verified_engine::verified]
pub(crate) fn rng_unit_f64(rng: &mut OxidizeRng) -> f64 {
    let val = (rng.next_u64() as f64) * INV_U64_MAX;
    debug_assert!((0.0..=1.0).contains(&val));
    assert!(val.is_finite());
    val
}

impl PhysarumEngine {
    #[verified_engine::verified]
    pub(crate) fn clear_obstacles_under_terminals(&mut self) {
        for idx in 0..self.terminal_count {
            let term = self.terminals[idx];
            let clear_r = (term.radius + 1.5).ceil() as isize;
            let tx = term.x.round() as isize;
            let ty = term.y.round() as isize;
            for dy in -clear_r..=clear_r {
                let ny = ty + dy;
                if ny < 0 || ny >= self.height as isize {
                    continue;
                }
                for dx in -clear_r..=clear_r {
                    let nx = tx + dx;
                    if nx >= 0
                        && nx < self.width as isize
                        && (dx as f64).hypot(dy as f64) <= term.radius + 1.5
                    {
                        let cell = flatten_2d_index(nx as usize, ny as usize, self.width);
                        self.obstacles[cell] = 0;
                    }
                }
            }
        }
        debug_assert!(self.terminal_count <= MAX_PHYSARUM_TERMINALS);
        assert_eq!(self.obstacles.len(), self.width * self.height);
    }

    #[verified_engine::verified]
    pub(crate) fn relocate_trapped_agents(&mut self) {
        if self.terminal_count == 0 {
            return;
        }
        for i in 0..self.agent_count {
            let gx = (self.agent_x[i] as usize).min(self.width - 1);
            let gy = (self.agent_y[i] as usize).min(self.height - 1);
            if self.obstacles[flatten_2d_index(gx, gy, self.width)] != 0 {
                let t_idx = i % self.terminal_count;
                let t = self.terminals[t_idx];
                self.agent_x[i] = t.x;
                self.agent_y[i] = t.y;
                self.agent_energy[i] = 1.0;
                self.agent_origin[i] = t_idx as u8;
            }
        }
        debug_assert!(self.agent_count <= MAX_PHYSARUM_AGENTS);
        assert!(self.terminal_count >= 1);
    }

    #[verified_engine::verified]
    pub(crate) fn emit_nutrient_beacons(&mut self) {
        let strength = self.params.beacon_strength;
        if strength <= 0.0 {
            return;
        }
        let target_cap = (2.2 * strength).min(MAX_TRAIL_CONCENTRATION * 0.55);
        for idx in 0..self.terminal_count {
            let term = self.terminals[idx];
            let span = term.radius.ceil() as isize;
            let cx = term.x.round() as isize;
            let cy = term.y.round() as isize;
            for dy in -span..=span {
                let ny = cy + dy;
                if ny < 1 || ny >= (self.height - 1) as isize {
                    continue;
                }
                for dx in -span..=span {
                    let nx = cx + dx;
                    let dist = (dx as f64).hypot(dy as f64);
                    if nx >= 1 && nx < (self.width - 1) as isize && dist <= term.radius {
                        let cell = flatten_2d_index(nx as usize, ny as usize, self.width);
                        if self.obstacles[cell] == 0 {
                            let kernel = 0.5 * (1.0 + (PI * dist / term.radius).cos());
                            let floor_val = target_cap * term.weight.min(2.0) * kernel;
                            let boosted = (self.trail[cell] + 0.35 * strength * kernel)
                                .max(floor_val)
                                .min(MAX_TRAIL_CONCENTRATION);
                            self.trail[cell] = boosted;
                        }
                    }
                }
            }
        }
        debug_assert!(strength.is_finite());
        assert!(self.terminal_count <= MAX_PHYSARUM_TERMINALS);
    }

    #[verified_engine::verified]
    fn sample_sensor(&self, sx: f64, sy: f64) -> f64 {
        if sx < 1.0
            || sy < 1.0
            || sx >= ((self.width - 1) as f64)
            || sy >= ((self.height - 1) as f64)
        {
            return -1.0;
        }
        let gx = sx as usize;
        let gy = sy as usize;
        let idx = flatten_2d_index(gx, gy, self.width);
        debug_assert!(idx < self.trail.len());
        assert!(gx < self.width && gy < self.height);
        if self.obstacles[idx] != 0 {
            -1.0
        } else {
            self.trail[idx]
        }
    }

    #[verified_engine::verified]
    fn steer_heading(&mut self, x: f64, y: f64, heading: f64) -> f64 {
        let sa = self.params.sensor_angle;
        let sd = self.params.sensor_distance;
        let ta = self.params.turn_angle;

        let fwd = self.sample_sensor(x + sd * heading.cos(), y + sd * heading.sin());
        let left_ang = heading - sa;
        let right_ang = heading + sa;
        let left = self.sample_sensor(x + sd * left_ang.cos(), y + sd * left_ang.sin());
        let right = self.sample_sensor(x + sd * right_ang.cos(), y + sd * right_ang.sin());

        let next = if fwd >= left && fwd >= right && fwd >= 0.0 {
            heading
        } else if fwd < left && fwd < right {
            if rng_unit_f64(&mut self.rng) < 0.5 {
                heading - ta
            } else {
                heading + ta
            }
        } else if left > right {
            heading - ta
        } else if right > left {
            heading + ta
        } else {
            heading + (rng_unit_f64(&mut self.rng) - 0.5) * 0.25 * ta
        };
        debug_assert!(next.is_finite());
        assert!(ta > 0.0);
        next.rem_euclid(TAU)
    }

    #[verified_engine::verified]
    fn best_departure_heading(&mut self, tx: f64, ty: f64, probe_r: f64) -> f64 {
        let mut best_val = -1.0;
        let mut best_ang = rng_unit_f64(&mut self.rng) * TAU;
        let phase = rng_unit_f64(&mut self.rng) * (TAU / 8.0);
        for dir in 0..8usize {
            let ang = phase + (dir as f64) * (TAU / 8.0);
            let val = self.sample_sensor(tx + probe_r * ang.cos(), ty + probe_r * ang.sin());
            if val > best_val {
                best_val = val;
                best_ang = ang;
            }
        }
        let jitter = (rng_unit_f64(&mut self.rng) - 0.5) * 0.32;
        debug_assert!(best_ang.is_finite());
        assert!(probe_r > 0.0);
        (best_ang + jitter).rem_euclid(TAU)
    }

    #[verified_engine::verified]
    fn feed_and_turn_at_terminal(&mut self, agent_idx: usize) {
        let x = self.agent_x[agent_idx];
        let y = self.agent_y[agent_idx];
        let prev_origin = self.agent_origin[agent_idx];
        for idx in 0..self.terminal_count {
            let term = self.terminals[idx];
            let dx = term.x - x;
            let dy = term.y - y;
            let r_feed = term.radius.max(2.6);
            if dx * dx + dy * dy <= r_feed * r_feed {
                self.agent_energy[agent_idx] = 1.0;
                if prev_origin != (idx as u8) {
                    self.agent_origin[agent_idx] = idx as u8;
                    let probe_r = (term.radius + 5.0).max(self.params.sensor_distance * 0.85);
                    self.agent_heading[agent_idx] =
                        self.best_departure_heading(term.x, term.y, probe_r);
                }
                debug_assert!(self.agent_energy[agent_idx] == 1.0);
                assert!(idx < MAX_PHYSARUM_TERMINALS);
                return;
            }
        }
    }

    #[verified_engine::verified]
    fn respawn_depleted_agent(&mut self, agent_idx: usize) {
        if self.terminal_count == 0 {
            return;
        }
        let pick = (self.rng.next_u64() as usize) % self.terminal_count;
        let term = self.terminals[pick];
        let probe_r = (term.radius + 5.0).max(self.params.sensor_distance * 0.85);
        let ang = self.best_departure_heading(term.x, term.y, probe_r);
        self.agent_x[agent_idx] = term.x;
        self.agent_y[agent_idx] = term.y;
        self.agent_heading[agent_idx] = ang;
        self.agent_energy[agent_idx] = 1.0;
        self.agent_origin[agent_idx] = pick as u8;
        debug_assert!(pick < self.terminal_count);
        assert!(self.agent_energy[agent_idx] > 0.0);
    }

    #[verified_engine::verified]
    pub(crate) fn step_soa_agents(&mut self) -> f64 {
        let speed = self.params.step_size;
        let dep = self.params.deposition;
        let max_x = (self.width - 1) as f64;
        let max_y = (self.height - 1) as f64;
        let mut moved = 0usize;

        for i in 0..self.agent_count {
            self.agent_energy[i] *= ENERGY_DECAY_PER_STEP;
            if self.agent_energy[i] < MIN_AGENT_ENERGY {
                self.respawn_depleted_agent(i);
            }
            self.feed_and_turn_at_terminal(i);

            let x = self.agent_x[i];
            let y = self.agent_y[i];
            let h = self.steer_heading(x, y, self.agent_heading[i]);
            let nx = x + speed * h.cos();
            let ny = y + speed * h.sin();

            if nx >= 1.0 && nx < max_x && ny >= 1.0 && ny < max_y {
                let cell = flatten_2d_index(nx as usize, ny as usize, self.width);
                if self.obstacles[cell] == 0 {
                    self.agent_x[i] = nx;
                    self.agent_y[i] = ny;
                    self.agent_heading[i] = h;
                    let flux_dep = dep * self.agent_energy[i];
                    self.trail[cell] = (self.trail[cell] + flux_dep).min(MAX_TRAIL_CONCENTRATION);
                    moved += 1;
                    continue;
                }
            }
            let bounce = (h + PI * (0.60 + 0.80 * rng_unit_f64(&mut self.rng))).rem_euclid(TAU);
            self.agent_heading[i] = bounce;
        }
        assert!(self.agent_count >= MIN_PHYSARUM_AGENTS);
        debug_assert!(moved <= self.agent_count);
        (moved as f64) / (self.agent_count as f64)
    }

    #[verified_engine::verified]
    pub(crate) fn diffuse_and_prune_trails(&mut self) {
        let alpha = self.params.diffusion_rate;
        let keep = 1.0 - self.params.decay_rate;
        let prune = self.params.prune_rate;
        let inv_tc_sq = 1.0 / (PRUNE_HALF_SATURATION * PRUNE_HALF_SATURATION);
        let w = self.width;
        let h = self.height;

        for y in 0..h {
            if y == 0 || y + 1 == h {
                for x in 0..w {
                    self.trail_scratch[flatten_2d_index(x, y, w)] = 0.0;
                }
                continue;
            }
            for x in 0..w {
                let center_idx = flatten_2d_index(x, y, w);
                if x == 0 || x + 1 == w || self.obstacles[center_idx] != 0 {
                    self.trail_scratch[center_idx] = 0.0;
                    continue;
                }
                let avg = self.mean_open_3x3(x, y, x - 1, x + 1, y - 1, y + 1);
                let diffused = (1.0 - alpha) * self.trail[center_idx] + alpha * avg;
                let hill_sink = prune * diffused / (1.0 + diffused * diffused * inv_tc_sq);
                self.trail_scratch[center_idx] =
                    (diffused * keep - hill_sink).clamp(0.0, MAX_TRAIL_CONCENTRATION);
            }
        }
        self.trail.copy_from_slice(&self.trail_scratch);
        debug_assert_eq!(self.trail.len(), self.trail_scratch.len());
        assert!(keep > 0.0);
    }

    #[verified_engine::verified]
    fn mean_open_3x3(
        &self,
        cx: usize,
        cy: usize,
        x0: usize,
        x1: usize,
        y0: usize,
        y1: usize,
    ) -> f64 {
        let center_val = self.trail[flatten_2d_index(cx, cy, self.width)];
        let mut acc = 0.0;
        for ny in [y0, cy, y1] {
            for nx in [x0, cx, x1] {
                let idx = flatten_2d_index(nx, ny, self.width);
                acc += if self.obstacles[idx] == 0 {
                    self.trail[idx]
                } else {
                    center_val
                };
            }
        }
        debug_assert!(acc.is_finite());
        assert!(acc >= 0.0);
        acc * (1.0 / 9.0)
    }

    #[verified_engine::verified]
    pub(crate) fn refresh_telemetry(&mut self, moved_fraction: f64) {
        let mut mass = 0.0;
        let mut active = 0.0;
        let mut peak = 0.0_f64;
        for idx in 0..self.trail.len() {
            if self.obstacles[idx] == 0 {
                let v = self.trail[idx];
                mass += v;
                if v >= ACTIVE_TUBE_THRESHOLD {
                    active += 1.0;
                }
                if v > peak {
                    peak = v;
                }
            }
        }
        self.telemetry.total_trail_mass = mass;
        self.telemetry.active_cells = active;
        self.telemetry.mean_flux = moved_fraction.clamp(0.0, 1.0);
        self.telemetry.peak_trail = peak;
        debug_assert!(mass.is_finite());
        assert!(peak >= 0.0);
    }
}
