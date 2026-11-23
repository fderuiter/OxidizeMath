//! Tactile brush operations and starter arrhythmia scenarios for [`super::sheet::ExcitableSheet2D`].

use super::sheet::{ExcitableSheet2D, ScheduledS2Pulse};
use math_commons::math_kernel::types::flatten_2d_index;

/// Canonical tissue excitation presets shared across simulation tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SheetPresetKind {
    /// Two planar waves launched from left and right edges that annihilate on head-on collision.
    DualEdgeCollision,
    /// Planar S1 wave followed by a timed S2 half-plane pulse inducing a self-sustaining spiral wave.
    S1S2SpiralInduction,
    /// Re-entrant rotor anchored around a central zero-flux ischemic scar obstacle.
    ScarPinnedRotor,
    /// Whole-sheet defibrillation shock restoring uniform resting equilibrium.
    WholeSheetDefibrillation,
}

impl ExcitableSheet2D {
    /// Paints transient electrode current `current` inside a disc centered at `(cx, cy)` with `radius`.
    #[verified_engine::verified]
    pub fn paint_current_disc(&mut self, cx: f64, cy: f64, radius: f64, current: f64) {
        let r = radius.max(0.5);
        let r_sq = r * r;
        let (x0, x1, y0, y1) = self.disc_bounds(cx, cy, r);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                if dx * dx + dy * dy <= r_sq {
                    let idx = flatten_2d_index(x, y, self.width());
                    if self.scar_mask()[idx] == 0 {
                        self.set_cell_stimulus(idx, current);
                    }
                }
            }
        }
    }

    /// Depolarizes excitable cells inside a disc to `target_u` (focal pulse stimulus).
    #[verified_engine::verified]
    pub fn pulse_voltage_disc(&mut self, cx: f64, cy: f64, radius: f64, target_u: f64) {
        let r = radius.max(0.5);
        let r_sq = r * r;
        let (x0, x1, y0, y1) = self.disc_bounds(cx, cy, r);
        let w = self.width();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                if dx * dx + dy * dy <= r_sq {
                    let idx = flatten_2d_index(x, y, w);
                    if self.scar_mask()[idx] == 0 {
                        self.voltage_mut()[idx] = target_u;
                    }
                }
            }
        }
    }

    /// Paints or clears a voltage-clamping probe inside a disc centered at `(cx, cy)`.
    #[verified_engine::verified]
    pub fn paint_voltage_clamp_disc(
        &mut self,
        cx: f64,
        cy: f64,
        radius: f64,
        target_u: f64,
        enabled: bool,
    ) {
        let r = radius.max(0.5);
        let r_sq = r * r;
        let (x0, x1, y0, y1) = self.disc_bounds(cx, cy, r);
        let w = self.width();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                if dx * dx + dy * dy <= r_sq {
                    let idx = flatten_2d_index(x, y, w);
                    self.set_cell_clamp(idx, target_u, enabled);
                }
            }
        }
    }

    /// Paints or erases a zero-flux insulating scar disc centered at `(cx, cy)`.
    #[verified_engine::verified]
    pub fn paint_scar_disc(&mut self, cx: f64, cy: f64, radius: f64, is_scar: bool) {
        let r = radius.max(0.5);
        let r_sq = r * r;
        let (x0, x1, y0, y1) = self.disc_bounds(cx, cy, r);
        let w = self.width();
        let (u_rest, w_rest) = self.system.kinetics.resting_state();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                if dx * dx + dy * dy <= r_sq {
                    let idx = flatten_2d_index(x, y, w);
                    self.scar_mask_mut()[idx] = u8::from(is_scar);
                    if is_scar {
                        self.voltage_mut()[idx] = u_rest;
                        self.recovery_mut()[idx] = w_rest;
                    }
                }
            }
        }
    }

    /// Paints a continuous scar or ablation line segment from `(x0, y0)` to `(x1, y1)`.
    #[verified_engine::verified]
    pub fn paint_scar_segment(
        &mut self,
        start: (f64, f64),
        end: (f64, f64),
        radius: f64,
        is_scar: bool,
    ) {
        let dist = ((end.0 - start.0).hypot(end.1 - start.1)).ceil() as usize;
        let steps = dist.max(1);
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let x = start.0 + (end.0 - start.0) * t;
            let y = start.1 + (end.1 - start.1) * t;
            self.paint_scar_disc(x, y, radius, is_scar);
        }
    }

    /// Applies one of the canonical starter scenarios to the tissue sheet.
    #[verified_engine::verified]
    pub fn apply_preset(&mut self, preset: SheetPresetKind) {
        match preset {
            SheetPresetKind::DualEdgeCollision => self.trigger_dual_edge_collision(),
            SheetPresetKind::S1S2SpiralInduction => self.trigger_s1_s2_spiral(),
            SheetPresetKind::ScarPinnedRotor => self.setup_scar_pinned_rotor(),
            SheetPresetKind::WholeSheetDefibrillation => self.defibrillate(),
        }
    }

    /// Launches opposing left-edge and right-edge planar wavefronts that annihilate upon collision.
    #[verified_engine::verified]
    pub fn trigger_dual_edge_collision(&mut self) {
        self.clear_scars();
        self.defibrillate();
        let w = self.width();
        let h = self.height();
        let band = (w / 16).max(3);
        for y in 0..h {
            for x in 0..band {
                let left = flatten_2d_index(x, y, w);
                let right = flatten_2d_index(w - 1 - x, y, w);
                self.voltage_mut()[left] = 1.8;
                self.voltage_mut()[right] = 1.8;
            }
        }
    }

    /// Launches a left-edge planar S1 wave and schedules a timed S2 pulse in its vulnerable refractory tail.
    #[verified_engine::verified]
    pub fn trigger_s1_s2_spiral(&mut self) {
        self.clear_scars();
        self.defibrillate();
        let w = self.width();
        let h = self.height();
        let band = (w / 16).max(4);
        for y in 0..h {
            for x in 0..band {
                let idx = flatten_2d_index(x, y, w);
                self.voltage_mut()[idx] = 1.85;
            }
        }
        let s2_delay = self.elapsed_time() + 82.0;
        self.set_pending_s2(Some(ScheduledS2Pulse {
            fire_time: s2_delay,
            x_min: 0,
            x_max: (w * 2) / 5,
            y_min: 0,
            y_max: h / 2,
            target_u: 1.9,
        }));
    }

    /// Initializes a central zero-flux ischemic scar with a unidirectional anchored re-entrant rotor.
    #[verified_engine::verified]
    pub fn setup_scar_pinned_rotor(&mut self) {
        self.clear_scars();
        self.defibrillate();
        let w = self.width();
        let h = self.height();
        let cx = w as f64 * 0.5;
        let cy = h as f64 * 0.5;
        let scar_radius = (w.min(h) as f64 * 0.24).max(6.0);
        self.paint_scar_disc(cx, cy, scar_radius, true);

        let mid_x = w / 2;
        let mid_y = h / 2;
        for y in 0..h {
            for x in mid_x..w {
                let idx = flatten_2d_index(x, y, w);
                if self.scar_mask()[idx] != 0 {
                    continue;
                }
                if y + 9 >= mid_y && y < mid_y {
                    self.voltage_mut()[idx] = 1.85;
                    self.recovery_mut()[idx] = -0.35;
                } else if y >= mid_y && y < mid_y + 18 {
                    self.voltage_mut()[idx] = -1.15;
                    self.recovery_mut()[idx] = 1.05;
                }
            }
        }
    }

    /// Connects the central scar (or sheet center) to the right boundary with an insulating ablation line.
    #[verified_engine::verified]
    pub fn ablate_scar_to_boundary(&mut self) {
        let cx = self.width() as f64 * 0.5;
        let cy = self.height() as f64 * 0.5;
        let edge_x = (self.width() - 1) as f64;
        self.paint_scar_segment((cx, cy), (edge_x, cy), 2.2, true);
    }

    /// Maximum membrane voltage currently present across all non-scar tissue cells.
    #[verified_engine::verified]
    pub fn max_tissue_voltage(&self) -> f64 {
        let (u_rest, _) = self.system.kinetics.resting_state();
        let mut max_u = u_rest;
        for (u, &scar) in self.voltage().iter().zip(self.scar_mask().iter()) {
            if scar == 0 && *u > max_u {
                max_u = *u;
            }
        }
        max_u
    }

    #[verified_engine::verified]
    fn disc_bounds(&self, cx: f64, cy: f64, radius: f64) -> (usize, usize, usize, usize) {
        let max_x = (self.width() - 1) as f64;
        let max_y = (self.height() - 1) as f64;
        let x0 = (cx - radius).floor().clamp(0.0, max_x) as usize;
        let x1 = (cx + radius).ceil().clamp(0.0, max_x) as usize;
        let y0 = (cy - radius).floor().clamp(0.0, max_y) as usize;
        let y1 = (cy + radius).ceil().clamp(0.0, max_y) as usize;
        (x0, x1, y0, y1)
    }

    #[verified_engine::verified]
    fn set_cell_stimulus(&mut self, idx: usize, current: f64) {
        self.stimulus[idx] = current;
    }

    #[verified_engine::verified]
    fn set_cell_clamp(&mut self, idx: usize, target_u: f64, enabled: bool) {
        if self.scar_mask[idx] != 0 {
            return;
        }
        self.clamp_mask[idx] = u8::from(enabled);
        if enabled {
            self.clamp_voltage[idx] = target_u;
            self.system.u_mut()[idx] = target_u;
        }
    }
}
