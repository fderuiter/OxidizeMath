//! Physics evaluation pipeline for `QuantumStatsPhononTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{QuantumStatsMetrics, QuantumStatsMode, QuantumStatsPhononTool};
use math_explorer::physics::solid_state::magnetism::{
    calculate_heisenberg_energy, magnon_dispersion,
};
use math_explorer::physics::solid_state::phonons::{
    acoustic_dispersion, debye_heat_capacity_low_temp, optical_dispersion,
};
use math_explorer::physics::solid_state::screening::{thomas_fermi_dielectric, yukawa_potential};
use math_explorer::physics::solid_state::second_quantization::{
    check_commutation, Boson, FockState, Operator, QuantumOperatorType,
};
use math_explorer::physics::solid_state::types::{ElectronVolts, Kelvin};
use math_explorer::physics::stat_mech::dynamics::estimate_diffusion_coefficient_with_rng;
use math_explorer::physics::stat_mech::ensembles::canonical_thermodynamics_ev;
use math_explorer::physics::stat_mech::quantum_stats::{
    calculate_occupancy_ev, BoseEinstein, FermiDirac, MaxwellBoltzmann,
};
use math_explorer::physics::stat_mech::KB;
use nalgebra::Vector3;
use oxidize_core::rng::OxidizeRng;

const ENERGY_SAMPLES: usize = 121;
const K_GRID_SAMPLES: usize = 81;
const DEBYE_SAMPLES: usize = 65;

impl QuantumStatsPhononTool {
    /// Evaluates quantum occupancies, canonical ensembles, phonons, and screening buffers.
    ///
    /// When `stats_mode == QuantumStatsMode::BoseEinstein` and $\mu \ge E$, records structured
    /// `StatMechError` guidance and preserves the last valid finite curve buffers for rollback.
    pub fn recalculate(&mut self) {
        let temp = Kelvin::new(self.temperature_k);
        let mu = ElectronVolts::new(self.chemical_potential_ev);

        if self.stats_mode == QuantumStatsMode::BoseEinstein {
            if let Err(err) = calculate_occupancy_ev(
                BoseEinstein,
                ElectronVolts::new(self.energy_min_ev),
                mu,
                temp,
            ) {
                let msg = format!(
                    "{err} — Bosonic chemical potential μ ({:.3} eV) must be strictly below ground state E_min ({:.3} eV) to avoid unphysical negative Bose occupancy outside a condensate.",
                    self.chemical_potential_ev, self.energy_min_ev
                );
                crate::accessibility::announce_status_with_priority(&msg, "assertive");
                self.stat_mech_error = Some(msg);
                return;
            }
        }
        self.stat_mech_error = None;

        self.sample_occupancy_curves(mu, temp);
        self.sample_phonons_magnons_and_debye();
        self.sample_screening_curves();
        self.metrics = self.compute_summary_metrics(temp);
        self.last_param_summary = self.param_summary();
    }

    fn sample_occupancy_curves(&mut self, mu: ElectronVolts, temp: Kelvin) {
        self.fd_points.clear();
        self.be_points.clear();
        self.mb_points.clear();
        let de = self.energy_span_ev / ((ENERGY_SAMPLES - 1) as f64);
        for i in 0..ENERGY_SAMPLES {
            let e_ev = self.energy_min_ev + (i as f64) * de;
            let e = ElectronVolts::new(e_ev);
            if let Ok(fd) = calculate_occupancy_ev(FermiDirac, e, mu, temp) {
                self.fd_points.push([e_ev, fd]);
            }
            if let Ok(mb) = calculate_occupancy_ev(MaxwellBoltzmann, e, mu, temp) {
                self.mb_points.push([e_ev, mb.min(8.0)]);
            }
            if e_ev > self.chemical_potential_ev + 1e-4 {
                if let Ok(be) = calculate_occupancy_ev(BoseEinstein, e, mu, temp) {
                    self.be_points.push([e_ev, be.min(8.0)]);
                }
            }
        }
    }

    fn sample_phonons_magnons_and_debye(&mut self) {
        self.acoustic_points.clear();
        self.optical_points.clear();
        self.magnon_points.clear();
        self.debye_cv_points.clear();

        let k_max = std::f64::consts::PI;
        let dk = k_max / ((K_GRID_SAMPLES - 1) as f64);
        for i in 0..K_GRID_SAMPLES {
            let k = (i as f64) * dk;
            let w_ac_mev = acoustic_dispersion(k, self.sound_velocity_eva);
            let w_op_mev = optical_dispersion(k, self.optical_omega_mev);
            let w_mag_mev = magnon_dispersion(k, self.exchange_j_mev, 0.5, 1.0);
            self.acoustic_points.push([k, w_ac_mev]);
            self.optical_points.push([k, w_op_mev]);
            self.magnon_points.push([k, w_mag_mev]);
        }

        let t_max = (0.35 * self.debye_theta_k).max(20.0);
        let dt = t_max / ((DEBYE_SAMPLES - 1) as f64);
        for i in 0..DEBYE_SAMPLES {
            let t_k = (i as f64) * dt;
            let cv_per_nk = debye_heat_capacity_low_temp(t_k, self.debye_theta_k, 1.0, KB) / KB;
            self.debye_cv_points.push([t_k, cv_per_nk]);
        }
    }

    fn sample_screening_curves(&mut self) {
        self.dielectric_points.clear();
        self.yukawa_points.clear();

        let (q_min, q_max) = (0.20, 4.0);
        let dq = (q_max - q_min) / ((K_GRID_SAMPLES - 1) as f64);
        let (r_min, r_max) = (0.25, 5.0);
        let dr = (r_max - r_min) / ((K_GRID_SAMPLES - 1) as f64);
        for i in 0..K_GRID_SAMPLES {
            let q = q_min + (i as f64) * dq;
            let eps = thomas_fermi_dielectric(q, self.k_tf_inv_a);
            self.dielectric_points.push([q, eps]);

            let r = r_min + (i as f64) * dr;
            let vr = yukawa_potential(r, self.k_tf_inv_a);
            self.yukawa_points.push([r, vr]);
        }
    }

    fn compute_summary_metrics(&self, temp: Kelvin) -> QuantumStatsMetrics {
        let mut ladder = [ElectronVolts::new(0.0); 16];
        let step = self.energy_span_ev / 15.0;
        for (idx, slot) in ladder.iter_mut().enumerate() {
            *slot = ElectronVolts::new(self.energy_min_ev + (idx as f64) * step);
        }
        let canon = canonical_thermodynamics_ev(&ladder, temp).unwrap_or(
            math_explorer::physics::stat_mech::ensembles::CanonicalThermodynamics {
                partition_function: 1.0,
                free_energy: ElectronVolts::new(self.energy_min_ev),
                internal_energy: ElectronVolts::new(self.energy_min_ev),
                entropy_ev_per_k: 0.0,
                entropy_j_per_k: 0.0,
            },
        );
        let debye_cv_per_nk =
            debye_heat_capacity_low_temp(self.temperature_k, self.debye_theta_k, 1.0, KB) / KB;
        let spins = [Vector3::new(0.0, 0.0, 0.5), Vector3::new(0.0, 0.0, 0.5)];
        let heisenberg_dimer_mev =
            calculate_heisenberg_energy(self.exchange_j_mev, &spins, &[(0, 1)]);
        let mut fock = FockState::new(1);
        let _ = fock.create_particle::<Boson>(0);
        let _ = fock.create_particle::<Boson>(0);
        let b_ann = Operator::new(QuantumOperatorType::Annihilation, 0);
        let b_cre = Operator::new(QuantumOperatorType::Creation, 0);
        let boson_ccr_norm =
            check_commutation::<Boson>(&b_ann, &b_cre) * (f64::from(fock.occupations[0]) * 0.5);
        let mut rng = OxidizeRng::from_seed(42);
        let lattice_diffusion_d = estimate_diffusion_coefficient_with_rng(64, 32, &mut rng);

        QuantumStatsMetrics {
            thermal_energy_mev: temp.thermal_energy_ev().as_f64() * 1e3,
            partition_z: canon.partition_function,
            free_energy_ev: canon.free_energy.as_f64(),
            internal_energy_ev: canon.internal_energy.as_f64(),
            entropy_mev_per_k: canon.entropy_ev_per_k * 1e3,
            debye_cv_per_nk,
            screening_length_a: 1.0 / self.k_tf_inv_a.max(1e-9),
            heisenberg_dimer_mev,
            boson_ccr_norm,
            lattice_diffusion_d,
        }
    }
}

// [cite:mmwave_radiotherapy_setup]
