//! # Zero-Allocation Symplectic Disk & Tidal Stream Orbits
//!
//! Implements a kick-drift-kick (KDK) symplectic leapfrog integrator for dwarf-galaxy
//! stellar disk tracers and eccentric tidal streams parameterized by [`GalaxyModel`]
//! and [`HaloProfile`]. All particle state and trail buffers are pre-allocated at
//! initialization so stepping inside [`verified_engine::engine::VerifiedEngine::run_verified`]
//! performs zero heap allocations while conserving energy and planar angular momentum.
//!
//! # Example
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::astrophysics::{
//!     GalacticDiskSystem, HaloProfile, Mpc, OrbitPopulationConfig, TypeCode10,
//! };
//!
//! let halo = HaloProfile::NfwCusp {
//!     virial_mass_solar: 1.0e10,
//!     scale_radius_kpc: 2.5,
//!     concentration: 12.0,
//! };
//! let mut sys = GalacticDiskSystem::from_galaxy_model(
//!     &TypeCode10,
//!     Mpc(10.0),
//!     0.6,
//!     0.8,
//!     halo,
//!     OrbitPopulationConfig::default(),
//! );
//! sys.step_n_leapfrog(2.0, 10);
//! assert!(sys.relative_energy_drift() < 1.0e-3);
//! ```

// [cite:dwarf_galaxy_empirical_dependencies]

use super::galaxies::{GalaxyModel, Mpc};
use super::rotation_curves::{
    BaryonicDiskConfig, DwarfMorphology, HaloProfile, METERS_PER_KPC,
    MIN_GALACTOCENTRIC_RADIUS_KPC, baryons_from_distance, decompose_rotation_curve,
    total_gravitational_potential,
};

/// Seconds per Megayear ($365.25 \times 86400 \times 10^6$).
pub const SECONDS_PER_MYR: f64 = 3.155_76e13;

/// Conversion factor from $\text{km/s}$ to $\text{kpc/Myr}$ ($\approx 1.022712 \times 10^{-3}$).
pub const KPC_PER_MYR_PER_KMS: f64 = (1_000.0 * SECONDS_PER_MYR) / METERS_PER_KPC;

/// Kinematic family of an orbiting test particle in the galactic midplane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OrbitParticleKind {
    /// Nearly circular disk star tracing differential rotation and spiral winding.
    #[default]
    DiskTracer,
    /// Eccentric satellite remnant tracing rosetta precession and tidal streaming.
    TidalStream,
}

/// Planar phase-space state $(x, y, v_x, v_y)$ of a massless tracer particle.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct OrbitalState {
    /// Galactocentric $x$ coordinate in kpc.
    pub x_kpc: f64,
    /// Galactocentric $y$ coordinate in kpc.
    pub y_kpc: f64,
    /// Planar $v_x$ velocity component in km/s.
    pub vx_kms: f64,
    /// Planar $v_y$ velocity component in km/s.
    pub vy_kms: f64,
    /// Tracer population tag.
    pub kind: OrbitParticleKind,
}

/// Instantaneous conserved quantities for a tracer particle in a static axisymmetric potential.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct OrbitalInvariants {
    /// Specific mechanical energy $E = \frac{1}{2}(v_x^2 + v_y^2) + \Phi(r)$ in $(\text{km/s})^2$.
    pub specific_energy_kms2: f64,
    /// Planar specific angular momentum $L_z = x v_y - y v_x$ in $\text{kpc}\cdot\text{km/s}$.
    pub angular_momentum_kpc_kms: f64,
    /// Galactocentric radius $r = \sqrt{x^2 + y^2}$ in kpc.
    pub radius_kpc: f64,
    /// Planar orbital speed $|\mathbf{v}| = \sqrt{v_x^2 + v_y^2}$ in km/s.
    pub speed_kms: f64,
}

/// Ensemble sizing and stream eccentricity parameters for [`GalacticDiskSystem`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OrbitPopulationConfig {
    /// Number of quasi-circular stellar disk tracer particles.
    pub disk_tracers: usize,
    /// Number of eccentric satellite tidal-stream tracer particles.
    pub tidal_tracers: usize,
    /// Pre-allocated ring-buffer capacity per particle trail.
    pub trail_capacity: usize,
    /// Ratio $v_{\text{tan}} / v_{\text{circ}}$ at apocenter for tidal stream tracers.
    pub tidal_tangential_ratio: f64,
}

impl Default for OrbitPopulationConfig {
    fn default() -> Self {
        Self {
            disk_tracers: 16,
            tidal_tracers: 6,
            trail_capacity: 48,
            tidal_tangential_ratio: 0.65,
        }
    }
}

/// Pre-allocated galactic disk and tidal stream orbit ensemble stepped via symplectic leapfrog.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GalacticDiskSystem {
    /// Baryonic stellar + HI gas disk derived from Tillaboev (2025).
    pub baryons: BaryonicDiskConfig,
    /// Active dark-matter halo profile.
    pub halo: HaloProfile,
    particles: Vec<OrbitalState>,
    trail_ring: Vec<[f64; 2]>,
    trail_capacity: usize,
    trail_head: usize,
    trail_len: usize,
    elapsed_myr: f64,
    initial_energy: f64,
    initial_lz: f64,
}

/// Computes planar central acceleration $(a_x, a_y)$ in $(\text{km/s})^2/\text{kpc}$ with softened core $r \ge 10^{-3}\,\text{kpc}$.
#[must_use]
#[verified_engine::verified]
pub fn planar_acceleration(
    x_kpc: f64,
    y_kpc: f64,
    baryons: &BaryonicDiskConfig,
    halo: &HaloProfile,
) -> (f64, f64) {
    let r_sq = x_kpc * x_kpc + y_kpc * y_kpc;
    let r_eff = r_sq.sqrt().max(MIN_GALACTOCENTRIC_RADIUS_KPC);
    let dec = decompose_rotation_curve(r_eff, baryons, halo);
    let omega_sq = (dec.v_circ_kms * dec.v_circ_kms) / (r_eff * r_eff);
    (-omega_sq * x_kpc, -omega_sq * y_kpc)
}

#[verified_engine::verified]
fn step_single_particle_kdk(
    p: &mut OrbitalState,
    ds_kpc_per_kms: f64,
    baryons: &BaryonicDiskConfig,
    halo: &HaloProfile,
) {
    let half_ds = 0.5 * ds_kpc_per_kms;
    let (ax0, ay0) = planar_acceleration(p.x_kpc, p.y_kpc, baryons, halo);
    let vx_half = p.vx_kms + half_ds * ax0;
    let vy_half = p.vy_kms + half_ds * ay0;
    let next_x = p.x_kpc + ds_kpc_per_kms * vx_half;
    let next_y = p.y_kpc + ds_kpc_per_kms * vy_half;
    let (ax1, ay1) = planar_acceleration(next_x, next_y, baryons, halo);
    p.x_kpc = next_x;
    p.y_kpc = next_y;
    p.vx_kms = vx_half + half_ds * ax1;
    p.vy_kms = vy_half + half_ds * ay1;
}

impl GalacticDiskSystem {
    /// Constructs a pre-allocated disk and tidal-stream ensemble from any [`GalaxyModel`] strategy.
    #[must_use]
    pub fn from_galaxy_model<M: GalaxyModel>(
        model: &M,
        distance: Mpc,
        gas_fraction: f64,
        disk_scale_kpc: f64,
        halo: HaloProfile,
        population: OrbitPopulationConfig,
    ) -> Self {
        let baryons = baryons_from_distance(model, distance, gas_fraction, disk_scale_kpc);
        Self::from_baryons(baryons, halo, population)
    }

    /// Constructs a pre-allocated disk and tidal-stream ensemble from a [`DwarfMorphology`] enum.
    #[must_use]
    pub fn from_morphology(
        morphology: DwarfMorphology,
        distance: Mpc,
        gas_fraction: f64,
        disk_scale_kpc: f64,
        halo: HaloProfile,
        population: OrbitPopulationConfig,
    ) -> Self {
        let baryons =
            morphology.baryonic_config_from_distance(distance, gas_fraction, disk_scale_kpc);
        Self::from_baryons(baryons, halo, population)
    }

    /// Constructs a pre-allocated system directly from [`BaryonicDiskConfig`] and [`HaloProfile`].
    #[must_use]
    pub fn from_baryons(
        baryons: BaryonicDiskConfig,
        halo: HaloProfile,
        population: OrbitPopulationConfig,
    ) -> Self {
        let disk_tracers = population.disk_tracers;
        let tidal_tracers = population.tidal_tracers;
        let tidal_tangential_ratio = population.tidal_tangential_ratio;
        let total = disk_tracers.saturating_add(tidal_tracers).max(1);
        let cap = population.trail_capacity.max(2);
        let mut sys = Self {
            baryons,
            halo,
            particles: vec![OrbitalState::default(); total],
            trail_ring: vec![[0.0, 0.0]; total.saturating_mul(cap)],
            trail_capacity: cap,
            trail_head: 0,
            trail_len: 0,
            elapsed_myr: 0.0,
            initial_energy: 0.0,
            initial_lz: 0.0,
        };
        sys.reseed_orbits(disk_tracers.max(1), tidal_tracers, tidal_tangential_ratio);
        sys
    }

    /// Reseeds initial tracer positions and velocities in place without reallocating when particle count matches.
    #[verified_engine::verified]
    pub fn reseed_orbits(
        &mut self,
        disk_tracers: usize,
        tidal_tracers: usize,
        tidal_tangential_ratio: f64,
    ) {
        let n_particles = self.particles.len();
        let disk_n = disk_tracers.min(n_particles);
        let tang_ratio = if tidal_tangential_ratio.is_finite() {
            tidal_tangential_ratio.clamp(0.15, 1.35)
        } else {
            0.65
        };
        self.seed_disk_tracers(disk_n);
        self.seed_tidal_tracers(disk_n, tidal_tracers, tang_ratio);
        self.trail_head = 0;
        self.trail_len = 0;
        self.elapsed_myr = 0.0;
        self.record_trail_snapshot();
        self.initial_energy = self.total_energy();
        self.initial_lz = self.total_angular_momentum();
    }

    #[verified_engine::verified]
    fn seed_disk_tracers(&mut self, disk_n: usize) {
        let r_base = self.baryons.disk_scale_kpc.max(0.25);
        let denom = disk_n.max(1) as f64;
        let mut i = 0;
        while i < disk_n {
            let u = (i as f64 + 0.5) / denom;
            let r = (0.55 + 5.2 * u) * r_base;
            // Align initial tracers along a two-arm bar/spiral phase to reveal differential winding
            let phi = if i % 2 == 0 {
                0.15 * u
            } else {
                std::f64::consts::PI + 0.15 * u
            };
            let v_circ = decompose_rotation_curve(r, &self.baryons, &self.halo).v_circ_kms;
            let (sin_p, cos_p) = phi.sin_cos();
            self.particles[i] = OrbitalState {
                x_kpc: r * cos_p,
                y_kpc: r * sin_p,
                vx_kms: -v_circ * sin_p,
                vy_kms: v_circ * cos_p,
                kind: OrbitParticleKind::DiskTracer,
            };
            i += 1;
        }
    }

    #[verified_engine::verified]
    fn seed_tidal_tracers(&mut self, start_idx: usize, tidal_tracers: usize, tang_ratio: f64) {
        let n_particles = self.particles.len();
        let r_apocenter = (4.5 * self.baryons.disk_scale_kpc).max(2.5);
        let count = tidal_tracers.min(n_particles.saturating_sub(start_idx));
        let denom = count.max(1) as f64;
        let mut j = 0;
        while j < count {
            let spread = (j as f64 / denom) - 0.5;
            let r = (r_apocenter * (1.0 + 0.14 * spread)).max(MIN_GALACTOCENTRIC_RADIUS_KPC);
            let angle = 0.6 + 0.18 * spread;
            let v_circ = decompose_rotation_curve(r, &self.baryons, &self.halo).v_circ_kms;
            let v_tan = v_circ * tang_ratio * (1.0 + 0.04 * spread);
            let (sin_a, cos_a) = angle.sin_cos();
            let idx = start_idx.saturating_add(j);
            self.particles[idx] = OrbitalState {
                x_kpc: r * cos_a,
                y_kpc: r * sin_a,
                vx_kms: -v_tan * sin_a,
                vy_kms: v_tan * cos_a,
                kind: OrbitParticleKind::TidalStream,
            };
            j += 1;
        }
    }

    #[verified_engine::verified]
    fn record_trail_snapshot(&mut self) {
        if self.trail_capacity == 0 || self.trail_ring.is_empty() {
            return;
        }
        let slot = self.trail_head;
        let cap = self.trail_capacity;
        let mut i = 0;
        while i < self.particles.len() {
            let base = i.saturating_mul(cap);
            let idx = base.saturating_add(slot);
            if idx < self.trail_ring.len() {
                self.trail_ring[idx] = [self.particles[i].x_kpc, self.particles[i].y_kpc];
            }
            i += 1;
        }
        self.trail_head = (slot + 1) % cap;
        if self.trail_len < cap {
            self.trail_len += 1;
        }
    }

    /// Advances all tracer orbits by a single KDK symplectic leapfrog step of duration `dt_myr` (Myr) with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_leapfrog(&mut self, dt_myr: f64) {
        if !dt_myr.is_finite() || dt_myr == 0.0 {
            return;
        }
        let clamped_dt = dt_myr.clamp(-25.0, 25.0);
        let ds = clamped_dt * KPC_PER_MYR_PER_KMS;
        let mut i = 0;
        while i < self.particles.len() {
            step_single_particle_kdk(&mut self.particles[i], ds, &self.baryons, &self.halo);
            i += 1;
        }
        self.elapsed_myr += clamped_dt;
        self.record_trail_snapshot();
    }

    /// Advances all tracer orbits by `substeps` consecutive KDK leapfrog steps of `dt_myr` each with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_n_leapfrog(&mut self, dt_myr: f64, substeps: usize) {
        let bounded = substeps.min(1_024);
        let mut s = 0;
        while s < bounded {
            self.step_leapfrog(dt_myr);
            s += 1;
        }
    }

    /// Executes [`Self::step_n_leapfrog`] inside [`verified_engine::engine::VerifiedEngine::run_verified`], returning `(function_calls, arithmetic_ops)`.
    pub fn step_n_verified(&mut self, dt_myr: f64, substeps: usize) -> (u64, u64) {
        let ((), metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            self.step_n_leapfrog(dt_myr, substeps);
        });
        (metrics.function_calls, metrics.arithmetic_ops)
    }

    /// Read-only slice of current particle states.
    #[must_use]
    #[verified_engine::verified]
    pub fn particles(&self) -> &[OrbitalState] {
        &self.particles
    }

    /// Elapsed simulation time in Megayears (Myr).
    #[must_use]
    #[verified_engine::verified]
    pub fn elapsed_myr(&self) -> f64 {
        self.elapsed_myr
    }

    /// Number of stored trail points per particle.
    #[must_use]
    #[verified_engine::verified]
    pub fn trail_len(&self) -> usize {
        self.trail_len
    }

    /// Copies the chronological `[x_kpc, y_kpc]` trail of particle `particle_idx` into `out`, returning the number of points written.
    #[verified_engine::verified]
    pub fn copy_particle_trail(&self, particle_idx: usize, out: &mut [[f64; 2]]) -> usize {
        if particle_idx >= self.particles.len() || self.trail_len == 0 || self.trail_capacity == 0 {
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

    /// Evaluates conserved orbital invariants for particle `index`.
    #[must_use]
    #[verified_engine::verified]
    pub fn particle_invariants(&self, index: usize) -> Option<OrbitalInvariants> {
        let p = self.particles.get(index)?;
        let r = (p.x_kpc * p.x_kpc + p.y_kpc * p.y_kpc).sqrt();
        let v_sq = p.vx_kms * p.vx_kms + p.vy_kms * p.vy_kms;
        let phi = total_gravitational_potential(r, &self.baryons, &self.halo);
        Some(OrbitalInvariants {
            specific_energy_kms2: 0.5 * v_sq + phi,
            angular_momentum_kpc_kms: p.x_kpc * p.vy_kms - p.y_kpc * p.vx_kms,
            radius_kpc: r,
            speed_kms: v_sq.sqrt(),
        })
    }

    /// Sum of specific mechanical energies across all tracer particles in $(\text{km/s})^2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_energy(&self) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        while i < self.particles.len() {
            if let Some(inv) = self.particle_invariants(i) {
                sum += inv.specific_energy_kms2;
            }
            i += 1;
        }
        sum
    }

    /// Sum of planar specific angular momenta $L_z$ across all tracer particles in $\text{kpc}\cdot\text{km/s}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_angular_momentum(&self) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        while i < self.particles.len() {
            let p = &self.particles[i];
            sum += p.x_kpc * p.vy_kms - p.y_kpc * p.vx_kms;
            i += 1;
        }
        sum
    }

    /// Dimensionless relative drift $|\Delta E| / \max(|E_0|, 10^{-12})$ from the initial ensemble energy.
    #[must_use]
    #[verified_engine::verified]
    pub fn relative_energy_drift(&self) -> f64 {
        let current = self.total_energy();
        (current - self.initial_energy).abs() / self.initial_energy.abs().max(1.0e-12)
    }

    /// Dimensionless relative drift $|\Delta L_z| / \max(|L_{z,0}|, 10^{-12})$ from the initial ensemble angular momentum.
    #[must_use]
    #[verified_engine::verified]
    pub fn relative_angular_momentum_drift(&self) -> f64 {
        let current = self.total_angular_momentum();
        (current - self.initial_lz).abs() / self.initial_lz.abs().max(1.0e-12)
    }
}
