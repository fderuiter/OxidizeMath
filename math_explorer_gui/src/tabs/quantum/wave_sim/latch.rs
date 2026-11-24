//! Born probability R/B/T integration and pre-wall scattering latch updates for [`super::WaveRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BARRIER_TRANSIT_THRESHOLD, TunnelingObservables, WALL_ARRIVAL_THRESHOLD, WaveRunner,
    compute_tunneling_observables_slice,
};

impl WaveRunner {
    /// Updates Born R/B/T integrals and pre-wall `T_scatter` latch in-place with zero heap allocations.
    pub(crate) fn update_observables_in_place(&mut self) {
        let n = self.psi.vector.len();
        if n <= self.prob_scratch.len() {
            self.psi
                .probability_density_in_place(&mut self.prob_scratch[..n]);
            let obs = compute_tunneling_observables_slice(
                &self.x_axis,
                &self.prob_scratch[..n],
                self.x_left,
                self.x_right,
            );
            let prev = self.observables;
            self.observables = obs;
            self.update_scatter_latch(prev, obs);
        }
    }

    fn update_scatter_latch(&mut self, prev: TunnelingObservables, obs: TunnelingObservables) {
        if self.time <= 0.0 {
            return;
        }
        let classical_past_barrier = (self.x0 + self.k0 * self.time) > (self.x_right + self.sigma);
        if obs.barrier_prob >= BARRIER_TRANSIT_THRESHOLD || classical_past_barrier {
            self.entered_barrier = true;
        }
        if self.entered_barrier && !self.wall_arrived {
            if obs.barrier_prob < BARRIER_TRANSIT_THRESHOLD {
                if obs.right_wall_prob < WALL_ARRIVAL_THRESHOLD {
                    self.latched_transmission = Some(obs.transmission);
                    self.quasi_bound_latched = false;
                } else if self.latched_transmission.is_none()
                    && prev.right_wall_prob < WALL_ARRIVAL_THRESHOLD
                {
                    let alpha_b = if prev.barrier_prob < BARRIER_TRANSIT_THRESHOLD {
                        0.0
                    } else {
                        let denom = (prev.barrier_prob - obs.barrier_prob).max(f64::EPSILON);
                        ((prev.barrier_prob - BARRIER_TRANSIT_THRESHOLD) / denom).clamp(0.0, 1.0)
                    };
                    let denom_w = (obs.right_wall_prob - prev.right_wall_prob).max(f64::EPSILON);
                    let alpha_w =
                        ((WALL_ARRIVAL_THRESHOLD - prev.right_wall_prob) / denom_w).clamp(0.0, 1.0);
                    if alpha_b <= alpha_w {
                        let t_pre =
                            prev.transmission + alpha_w * (obs.transmission - prev.transmission);
                        self.latched_transmission = Some(t_pre.clamp(0.0, 1.0));
                        self.quasi_bound_latched = false;
                    }
                }
            } else if self.latched_transmission.is_none()
                && obs.right_wall_prob >= WALL_ARRIVAL_THRESHOLD
            {
                let alpha_w = if prev.right_wall_prob < WALL_ARRIVAL_THRESHOLD {
                    let denom_w = (obs.right_wall_prob - prev.right_wall_prob).max(f64::EPSILON);
                    ((WALL_ARRIVAL_THRESHOLD - prev.right_wall_prob) / denom_w).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let t_pre = prev.transmission + alpha_w * (obs.transmission - prev.transmission);
                self.latched_transmission = Some(t_pre.clamp(0.0, 1.0));
                self.quasi_bound_latched = true;
            }
        }
        if obs.right_wall_prob >= WALL_ARRIVAL_THRESHOLD
            || (self.latched_transmission.is_some() && obs.left_wall_prob >= WALL_ARRIVAL_THRESHOLD)
        {
            self.wall_arrived = true;
        }
    }
}
