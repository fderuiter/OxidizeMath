//! Physics evaluation pipeline for `BcsSuperconductivityTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{BcsSummaryMetrics, BcsSuperconductivityTool};
use math_explorer::physics::solid_state::bcs::{
    coherence_factors_normal_limit, dynes_dos_ratio, nis_tunneling_conductance,
    quasiparticle_energy, BcsGapSolver, IsotropicBCSModel,
};
use math_explorer::physics::solid_state::interactions::FrohlichVertex;
use math_explorer::physics::solid_state::second_quantization::{
    check_commutation, Fermion, FockState, Operator, QuantumOperatorType,
};
use math_explorer::physics::solid_state::types::{ElectronVolts, Kelvin};

const BCS_DISCRETE_LEVELS: usize = 160;
const GAP_TEMP_SAMPLES: usize = 65;
const DISPERSION_SAMPLES: usize = 121;
const NIS_BIAS_SAMPLES: usize = 121;
const FROHLICH_Q_SAMPLES: usize = 65;

impl BcsSuperconductivityTool {
    /// Solves the finite-temperature BCS gap equation and refreshes all spectroscopy buffers.
    pub fn recalculate(&mut self) {
        self.solver_error = None;
        let cutoff_ev = self.debye_energy_mev * 1e-3;
        let solver = BcsGapSolver::new()
            .with_tolerance(1e-11)
            .with_max_iterations(250);
        let Some((base_model, tc_kelvin, delta_zero_ev, delta_t_ev)) =
            self.solve_operating_gaps(cutoff_ev, &solver)
        else {
            return;
        };

        self.sample_gap_vs_temperature(&base_model, &solver, tc_kelvin);
        let ref_scale_mev = (delta_zero_ev * 1e3).max(0.5);
        self.sample_quasiparticles(delta_t_ev, ref_scale_mev);
        let zero_bias_g = self.sample_nis_and_frohlich(delta_t_ev, ref_scale_mev);
        let kt_c_ev = Kelvin::new(tc_kelvin).thermal_energy_ev().as_f64();
        let universal_ratio = if kt_c_ev > 1e-15 {
            (2.0 * delta_zero_ev) / kt_c_ev
        } else {
            0.0
        };
        let frohlich_ref = FrohlichVertex::new(self.frohlich_alpha, cutoff_ev)
            .amplitude(1.5)
            .norm_sqr();

        self.metrics = BcsSummaryMetrics {
            delta_zero_mev: delta_zero_ev * 1e3,
            delta_t_mev: delta_t_ev * 1e3,
            tc_kelvin,
            universal_ratio,
            zero_bias_conductance: zero_bias_g,
            frohlich_mq_sq_ev2: frohlich_ref,
            cooper_car_norm: evaluate_cooper_car_norm(),
        };
        self.last_param_summary = self.param_summary();
    }

    fn solve_operating_gaps(
        &mut self,
        cutoff_ev: f64,
        solver: &BcsGapSolver,
    ) -> Option<(IsotropicBCSModel, f64, f64, f64)> {
        let d_xi = (2.0 * cutoff_ev) / (BCS_DISCRETE_LEVELS as f64);
        let energies: Vec<f64> = (0..BCS_DISCRETE_LEVELS)
            .map(|i| -cutoff_ev + ((i as f64) + 0.5) * d_xi)
            .collect();
        let base = IsotropicBCSModel::new(energies, self.coupling_nv * d_xi, cutoff_ev)
            .map_err(|e| self.solver_error = Some(e.to_string()))
            .ok()?;
        let tc_k = base
            .critical_temperature()
            .map_err(|e| self.solver_error = Some(e.to_string()))
            .ok()?
            .as_f64();
        let d0 = solver
            .solve(&base, ElectronVolts::new(0.0))
            .map_err(|e| self.solver_error = Some(e.to_string()))
            .ok()?
            .as_f64();
        let op = base
            .clone()
            .with_temperature(Kelvin::new(self.temperature_k));
        let dt = solver
            .solve(&op, ElectronVolts::new(0.0))
            .map_err(|e| self.solver_error = Some(e.to_string()))
            .ok()?
            .as_f64();
        Some((base, tc_k, d0, dt))
    }

    fn sample_gap_vs_temperature(
        &mut self,
        base_model: &IsotropicBCSModel,
        solver: &BcsGapSolver,
        tc_kelvin: f64,
    ) {
        self.gap_vs_temp_points.clear();
        let t_max = (1.35 * tc_kelvin).max(self.temperature_k * 1.1).max(1.0);
        let dt = t_max / ((GAP_TEMP_SAMPLES - 1) as f64);
        for i in 0..GAP_TEMP_SAMPLES {
            let t_k = (i as f64) * dt;
            let model_t = base_model.clone().with_temperature(Kelvin::new(t_k));
            if let Ok(gap) = solver.solve(&model_t, ElectronVolts::new(0.0)) {
                self.gap_vs_temp_points.push([t_k, gap.as_f64() * 1e3]);
            }
        }
    }

    fn sample_quasiparticles(&mut self, delta_t_ev: f64, ref_scale_mev: f64) {
        self.dispersion_points.clear();
        self.normal_disp_points.clear();
        self.u_sq_points.clear();
        self.v_sq_points.clear();
        let xi_span_mev = 3.0 * ref_scale_mev;
        let d_xi = (2.0 * xi_span_mev) / ((DISPERSION_SAMPLES - 1) as f64);
        let delta = ElectronVolts::new(delta_t_ev);
        for i in 0..DISPERSION_SAMPLES {
            let xi_mev = -xi_span_mev + (i as f64) * d_xi;
            let xi = ElectronVolts::new(xi_mev * 1e-3);
            let e_mev = quasiparticle_energy(xi, delta).as_f64() * 1e3;
            let (u, v) = coherence_factors_normal_limit(xi, delta).unwrap_or((1.0, 0.0));
            self.dispersion_points.push([xi_mev, e_mev]);
            self.normal_disp_points.push([xi_mev, xi_mev.abs()]);
            self.u_sq_points.push([xi_mev, u * u]);
            self.v_sq_points.push([xi_mev, v * v]);
        }
    }

    fn sample_nis_and_frohlich(&mut self, delta_t_ev: f64, ref_scale_mev: f64) -> f64 {
        self.nis_didv_points.clear();
        self.dynes_dos_points.clear();
        self.frohlich_points.clear();

        let v_span_mev = 3.5 * ref_scale_mev;
        let dv = (2.0 * v_span_mev) / ((NIS_BIAS_SAMPLES - 1) as f64);
        let delta = ElectronVolts::new(delta_t_ev);
        let gamma = ElectronVolts::new(self.dynes_gamma_mev * 1e-3);
        let temp = Kelvin::new(self.temperature_k);

        for i in 0..NIS_BIAS_SAMPLES {
            let bias_mev = -v_span_mev + (i as f64) * dv;
            let bias = ElectronVolts::new(bias_mev * 1e-3);
            let didv = nis_tunneling_conductance(bias, delta, gamma, temp).unwrap_or(1.0);
            let dos = dynes_dos_ratio(bias, delta, gamma).unwrap_or(1.0);
            self.nis_didv_points.push([bias_mev, didv]);
            self.dynes_dos_points.push([bias_mev, dos]);
        }

        let vertex = FrohlichVertex::new(self.frohlich_alpha, self.debye_energy_mev * 1e-3);
        let (q_min, q_max) = (0.2, 3.0);
        let dq = (q_max - q_min) / ((FROHLICH_Q_SAMPLES - 1) as f64);
        for i in 0..FROHLICH_Q_SAMPLES {
            let q = q_min + (i as f64) * dq;
            self.frohlich_points
                .push([q, vertex.amplitude(q).norm_sqr()]);
        }

        nis_tunneling_conductance(ElectronVolts::new(0.0), delta, gamma, temp).unwrap_or(1.0)
    }
}

fn evaluate_cooper_car_norm() -> f64 {
    let mut fock = FockState::new(2);
    let _ = fock.create_particle::<Fermion>(0);
    let _ = fock.create_particle::<Fermion>(1);
    let c_ann = Operator::new(QuantumOperatorType::Annihilation, 0);
    let c_cre = Operator::new(QuantumOperatorType::Creation, 0);
    let pair_occ = f64::from(fock.occupations.iter().copied().sum::<u8>()) * 0.5;
    check_commutation::<Fermion>(&c_ann, &c_cre) * pair_occ
}

// [cite:mmwave_radiotherapy_setup]
