// [cite:optics]

use super::{WaveField2D, WaveSource2D};
use math_commons::math_kernel::types::flatten_2d_index;

#[derive(Clone, Copy)]
struct DrudeStepCoeffs {
    w: usize,
    h: usize,
    ds: f64,
    g: f64,
    decay_m: f64,
    inv_dx: f64,
    inv_dy: f64,
}

impl WaveField2D {
    #[verified_engine::verified]
    pub(super) fn step_drude_metamaterial(&mut self, dt: f64, source: Option<&WaveSource2D>) {
        let w = self.config.width;
        let h = self.config.height;
        let ds = self.config.c0 * dt;
        let k0 = self.drude_omega_ref / self.config.c0;
        let gamma = 0.015 * k0;
        let g = 1.0 / (1.0 + 0.5 * gamma * ds);
        let coeffs = DrudeStepCoeffs {
            w,
            h,
            ds,
            g,
            decay_m: 1.0 - 0.5 * gamma * ds,
            inv_dx: self.ops.inv_dx,
            inv_dy: 1.0 / *self.ops.dy,
        };
        self.update_drude_magnetic(coeffs);
        let src_val = source.map_or(0.0, |s| s.evaluate_at_time(self.time));
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = flatten_2d_index(x, y, w);
                if self.obstacle[idx] != 0 {
                    self.u_curr[idx] = 0.0;
                    self.jz[idx] = 0.0;
                    continue;
                }
                let n_val = self.refractive_index[idx];
                let eps_inf = if n_val > 0.0 { n_val * n_val } else { 1.0 };
                let kp_sq = self.omega_p_sq[idx];
                let sigma = self.damping[idx] / self.config.c0;
                let curl_h = (self.hy[idx] - self.hy[idx - 1]) * coeffs.inv_dx
                    - (self.hx[idx] - self.hx[idx - w]) * coeffs.inv_dy;
                let s_term = Self::source_weight(source, x, y, src_val);
                let beta = 0.5 * ds * (sigma * eps_inf + 0.5 * ds * g * kp_sq);
                let u_old = self.u_curr[idx];
                let j_old = self.jz[idx];
                let u_new = ((eps_inf - beta) * u_old + ds * (curl_h - g * j_old + s_term))
                    / (eps_inf + beta);
                self.u_curr[idx] = u_new;
                self.jz[idx] = g * (coeffs.decay_m * j_old + 0.5 * ds * kp_sq * (u_new + u_old));
            }
        }
    }

    #[verified_engine::verified]
    fn update_drude_magnetic(&mut self, c: DrudeStepCoeffs) {
        let inv_c0 = 1.0 / self.config.c0;
        for y in 0..c.h - 1 {
            for x in 0..c.w - 1 {
                let idx = flatten_2d_index(x, y, c.w);
                let kp_y = 0.5 * (self.omega_p_sq[idx] + self.omega_p_sq[idx + c.w]);
                let sig_y = 0.5 * (self.damping[idx] + self.damping[idx + c.w]) * inv_c0;
                let beta_y = 0.5 * c.ds * (sig_y + 0.5 * c.ds * c.g * kp_y);
                let hx_old = self.hx[idx];
                let mx_old = self.mx[idx];
                let du_dy = (self.u_curr[idx + c.w] - self.u_curr[idx]) * c.inv_dy;
                let hx_new =
                    ((1.0 - beta_y) * hx_old - c.ds * (du_dy + c.g * mx_old)) / (1.0 + beta_y);
                self.hx[idx] = hx_new;
                self.mx[idx] = c.g * (c.decay_m * mx_old + 0.5 * c.ds * kp_y * (hx_new + hx_old));

                let kp_x = 0.5 * (self.omega_p_sq[idx] + self.omega_p_sq[idx + 1]);
                let sig_x = 0.5 * (self.damping[idx] + self.damping[idx + 1]) * inv_c0;
                let beta_x = 0.5 * c.ds * (sig_x + 0.5 * c.ds * c.g * kp_x);
                let hy_old = self.hy[idx];
                let my_old = self.my[idx];
                let du_dx = (self.u_curr[idx + 1] - self.u_curr[idx]) * c.inv_dx;
                let hy_new =
                    ((1.0 - beta_x) * hy_old + c.ds * (du_dx - c.g * my_old)) / (1.0 + beta_x);
                self.hy[idx] = hy_new;
                self.my[idx] = c.g * (c.decay_m * my_old + 0.5 * c.ds * kp_x * (hy_new + hy_old));
            }
        }
    }
}
