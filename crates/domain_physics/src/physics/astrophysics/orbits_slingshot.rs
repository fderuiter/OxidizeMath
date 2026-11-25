//! Zero-allocation canvas slingshot tracer injection and galactic surface-brightness sampling.

use super::orbits::{GalacticDiskSystem, OrbitParticleKind, OrbitalState};
use super::paczynski_wiita::paczynski_wiita_potential_kpc;
use super::rotation_curves::total_gravitational_potential;

impl GalacticDiskSystem {
    /// Injects or overwrites a [`OrbitParticleKind::SlingshotProbe`] in the pre-allocated ring
    /// slots of [`GalacticDiskSystem`] with zero heap allocations, returning the particle index.
    ///
    /// Updates ensemble baseline invariants (`initial_energy` and `initial_lz`) by the exact
    /// single-particle delta so subsequent [`Self::relative_energy_drift`] measurements reflect
    /// genuine symplectic integrator accuracy.
    #[verified_engine::verified]
    pub fn launch_tracer_in_place(
        &mut self,
        x_kpc: f64,
        y_kpc: f64,
        vx_kms: f64,
        vy_kms: f64,
    ) -> usize {
        if self.particles.is_empty() {
            return 0;
        }
        let clean_x = if x_kpc.is_finite() {
            x_kpc.clamp(-40.0, 40.0)
        } else {
            2.0
        };
        let clean_y = if y_kpc.is_finite() {
            y_kpc.clamp(-40.0, 40.0)
        } else {
            0.0
        };
        let clean_vx = if vx_kms.is_finite() {
            vx_kms.clamp(-800.0, 800.0)
        } else {
            0.0
        };
        let clean_vy = if vy_kms.is_finite() {
            vy_kms.clamp(-800.0, 800.0)
        } else {
            40.0
        };

        let base = if self.base_count == 0 {
            self.particles.len()
        } else {
            self.base_count.min(self.particles.len())
        };
        let ring_cap = self.particles.len().saturating_sub(base);
        let slot = if ring_cap > 0 {
            base + (self.slingshot_cursor % ring_cap)
        } else {
            self.particles.len() - 1
        };

        let active_before = self.effective_active_len();
        if slot < active_before {
            let (old_e, old_lz) = self.single_slot_energy_lz(slot);
            self.initial_energy -= old_e;
            self.initial_lz -= old_lz;
        }

        self.particles[slot] = OrbitalState {
            x_kpc: clean_x,
            y_kpc: clean_y,
            vx_kms: clean_vx,
            vy_kms: clean_vy,
            kind: OrbitParticleKind::SlingshotProbe,
            captured: false,
        };
        self.reset_slot_trail(slot, [clean_x, clean_y]);

        if ring_cap > 0 {
            self.slingshot_cursor = (self.slingshot_cursor + 1) % ring_cap;
        }
        self.active_count = active_before.max(slot + 1);

        let (new_e, new_lz) = self.single_slot_energy_lz(slot);
        self.initial_energy += new_e;
        self.initial_lz += new_lz;
        slot
    }

    /// Number of currently active [`OrbitParticleKind::SlingshotProbe`] tracers in the ensemble.
    #[must_use]
    #[verified_engine::verified]
    pub fn slingshot_probe_count(&self) -> usize {
        let active = self.particles();
        let mut count = 0;
        let mut i = 0;
        while i < active.len() {
            if active[i].kind == OrbitParticleKind::SlingshotProbe {
                count += 1;
            }
            i += 1;
        }
        count
    }

    /// Evaluates normalized RGB `[r, g, b]` surface brightness (`[0.0, 1.0]`) of the exponential
    /// stellar disk, logarithmic spiral arms, HI gas ring, and orbiting tracers at `(x_kpc, y_kpc)`
    /// with zero heap allocations for cross-domain gravitational lensing.
    #[must_use]
    #[verified_engine::verified]
    pub fn sample_surface_brightness(&self, x_kpc: f64, y_kpc: f64) -> [f64; 3] {
        if !x_kpc.is_finite() || !y_kpc.is_finite() {
            return [0.0, 0.0, 0.0];
        }
        let r_d = self.baryons.disk_scale_kpc.max(0.25);
        let r = (x_kpc * x_kpc + y_kpc * y_kpc).sqrt();
        let phi = y_kpc.atan2(x_kpc);
        let bulge = (-2.2 * (r / r_d).powi(2)).exp();
        let exp_disk = (-r / (1.8 * r_d)).exp();
        let omega_phase = 0.015 * self.elapsed_myr;
        let spiral_arg = 2.0 * (phi - omega_phase) - 3.4 * ((r + 0.15 * r_d) / r_d).ln();
        let arm_wave = 0.5 + 0.5 * spiral_arg.cos();
        let gas_ring = (-(r - 2.4 * r_d).powi(2) / (2.5 * r_d * r_d)).exp() * arm_wave;

        let mut star_glow = 0.0_f64;
        if r < 7.0 * r_d {
            let active = self.particles();
            let mut i = 0;
            while i < active.len() {
                let dx = x_kpc - active[i].x_kpc;
                let dy = y_kpc - active[i].y_kpc;
                let d2 = dx * dx + dy * dy;
                if d2 < 0.35 * r_d * r_d {
                    star_glow += (-d2 / (0.045 * r_d * r_d)).exp() * 0.45;
                }
                i += 1;
            }
        }

        let warm =
            (0.95 * bulge + 0.55 * exp_disk * (0.45 + 0.55 * arm_wave) + star_glow).clamp(0.0, 1.0);
        let total_baryon_mass =
            (self.baryons.disk_mass_solar + self.baryons.gas_mass_solar).max(1.0);
        let gas_frac = (self.baryons.gas_mass_solar / total_baryon_mass).clamp(0.15, 1.0);
        let cyan_gas = (0.65 * gas_ring * gas_frac + 0.35 * star_glow).clamp(0.0, 1.0);
        [
            warm,
            (0.76 * warm + 0.48 * cyan_gas).clamp(0.0, 1.0),
            (0.45 * warm + 0.92 * cyan_gas).clamp(0.0, 1.0),
        ]
    }

    #[verified_engine::verified]
    pub(super) fn copy_particle_trail_inner(
        &self,
        particle_idx: usize,
        out: &mut [[f64; 2]],
    ) -> usize {
        if particle_idx >= self.effective_active_len()
            || self.trail_len == 0
            || self.trail_capacity == 0
        {
            return 0;
        }
        let count = self.trail_len.min(out.len());
        let cap = self.trail_capacity;
        let start_slot = if self.trail_len < cap {
            0
        } else {
            self.trail_head
        };
        let base = particle_idx.saturating_mul(cap);
        let mut k = 0;
        while k < count {
            let slot = (start_slot + (self.trail_len - count) + k) % cap;
            let idx = base.saturating_add(slot);
            if idx < self.trail_ring.len() {
                out[k] = self.trail_ring[idx];
            }
            k += 1;
        }
        count
    }

    #[verified_engine::verified]
    fn single_slot_energy_lz(&self, slot: usize) -> (f64, f64) {
        let p = &self.particles[slot];
        let r = (p.x_kpc * p.x_kpc + p.y_kpc * p.y_kpc).sqrt();
        let v_sq = p.vx_kms * p.vx_kms + p.vy_kms * p.vy_kms;
        let phi = total_gravitational_potential(r, &self.baryons, &self.halo)
            + paczynski_wiita_potential_kpc(r, &self.central_bh);
        (0.5 * v_sq + phi, p.x_kpc * p.vy_kms - p.y_kpc * p.vx_kms)
    }

    #[verified_engine::verified]
    fn reset_slot_trail(&mut self, slot: usize, pos: [f64; 2]) {
        let cap = self.trail_capacity;
        let base = slot.saturating_mul(cap);
        let mut k = 0;
        while k < cap {
            let idx = base.saturating_add(k);
            if idx < self.trail_ring.len() {
                self.trail_ring[idx] = pos;
            }
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use verified_engine::allocator::{lock_allocations, unlock_allocations};
    use verified_engine::metrics::{disable_verification, enable_verification};

    use crate::physics::astrophysics::{
        GalacticDiskSystem, HaloProfile, Mpc, OrbitParticleKind, OrbitPopulationConfig,
        SLINGSHOT_RING_CAPACITY, TypeCode10,
    };

    #[test]
    #[verified_engine::verified]
    fn test_launch_tracer_in_place_zero_alloc_ring_wrap_and_energy_conservation() {
        let halo = HaloProfile::BurkertCore {
            virial_mass_solar: 1.2e10,
            core_radius_kpc: 2.2,
            concentration: 11.0,
        };
        let mut sys = GalacticDiskSystem::from_galaxy_model(
            &TypeCode10,
            Mpc(12.0),
            0.6,
            0.8,
            halo,
            OrbitPopulationConfig::default(),
        );
        let base_len = sys.particles().len();
        assert_eq!(base_len, 22);
        assert_eq!(sys.slingshot_probe_count(), 0);

        enable_verification();
        lock_allocations();
        for k in 0..(SLINGSHOT_RING_CAPACITY + 4) {
            let angle = (k as f64) * 0.45;
            let idx = sys.launch_tracer_in_place(2.4 * angle.cos(), 2.4 * angle.sin(), -18.0, 46.0);
            assert!(idx >= base_len && idx < base_len + SLINGSHOT_RING_CAPACITY);
        }
        let mut brightness_sum = 0.0;
        for step in 0..1_000 {
            sys.step_leapfrog(0.5);
            let rgb = sys.sample_surface_brightness(1.2, (step % 7) as f64 * 0.25);
            brightness_sum += rgb[0] + rgb[1] + rgb[2];
        }
        unlock_allocations();
        disable_verification();

        assert_eq!(sys.slingshot_probe_count(), SLINGSHOT_RING_CAPACITY);
        assert!(
            sys.particles()[base_len..]
                .iter()
                .all(|p| p.kind == OrbitParticleKind::SlingshotProbe)
        );
        assert!(sys.relative_energy_drift() < 1.0e-3);
        assert!(sys.relative_angular_momentum_drift() < 1.0e-9);
        assert!(brightness_sum > 10.0);
    }
}
