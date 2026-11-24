#![cfg_attr(any(), verified(opt_out = "verification_harness"))]

use crate::mms::{SolverCertificate, ols_convergence_order};
use domain_physics::physics::astrophysics::nbody::FigureEightThreeBody;
use domain_physics::physics::astrophysics::{
    DwarfMorphology, GalacticDiskSystem, HaloProfile, Mpc, OrbitPopulationConfig,
};
use domain_physics::physics::chaos::double_pendulum::DoublePendulum;
use domain_physics::physics::chaos::henon_heiles::HenonHeiles;
use domain_physics::physics::chaos::lorenz::{LorenzState, LorenzSystem};
use domain_physics::physics::fluid_dynamics::conservation::Euler as EulerMomentum;
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::{
    BgkCollision, D2Q9, Lattice2D, LatticeBoltzmannD2Q9,
};
use domain_physics::physics::fluid_dynamics::solver::FluidParticleSystem;
use domain_physics::physics::fluid_dynamics::types::{
    FlowState, FluidProperties, SpatialGradients,
};
use domain_physics::physics::high_energy::general_relativity::SchwarzschildGeodesicBundle;
use domain_physics::physics::mri::bloch::BlochSimulator;
use domain_physics::physics::optics::{
    EikonalRaySystem, GrinMedium, HamiltonianRayTracer, OpticalMediumConfig, OpticsPreset,
};
use domain_physics::physics::quantum::schrodinger::time_evolution_operator;
use domain_physics::physics::quantum::tridiagonal::{
    CrankNicolsonPropagator1D, TridiagonalHamiltonian1D,
};
use domain_physics::physics::quantum::types::{QuantumOperator, QuantumState};
use domain_physics::physics::solid_state::bcs::{BcsGapSolver, IsotropicBCSModel};
use domain_physics::physics::solid_state::types::ElectronVolts;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use num_complex::Complex;
use pure_math::pure_math::analysis::ode::symplectic::Yoshida4;
use pure_math::pure_math::analysis::ode::{ArrayState, RungeKutta4, Solver, TimeStepper};
use std::f64::consts::PI;

fn phys_cert(
    name: &'static str,
    min_p: f64,
    hs: &[f64],
    errs: &[f64],
    max_err: f64,
) -> SolverCertificate {
    SolverCertificate::evaluate(
        name,
        "domain_physics",
        min_p,
        ols_convergence_order(hs, errs),
        errs[2],
        max_err,
    )
}

fn certify_astrophysics() -> Vec<SolverCertificate> {
    let nbody = FigureEightThreeBody::default();
    let s0_nb = FigureEightThreeBody::figure_eight_initial_state();
    let mut ref_nb = s0_nb;
    let mut yo_ref = Yoshida4::new(&ArrayState([0.0; 6]));
    for _ in 0..800 {
        yo_ref.step_separable(&nbody, &mut ref_nb, 0.05 / 800.0);
    }
    let (mut hs_nb, mut err_nb) = ([0.0; 3], [0.0; 3]);
    for (k, &n) in [10usize, 20, 40].iter().enumerate() {
        let dt = 0.05 / n as f64;
        hs_nb[k] = dt;
        let (mut s, mut yo) = (s0_nb, Yoshida4::new(&ArrayState([0.0; 6])));
        for _ in 0..n {
            yo.step_separable(&nbody, &mut s, dt);
        }
        err_nb[k] = s.max_abs_diff(&ref_nb);
    }

    let eval_gal = |sub: usize| -> [f64; 2] {
        let mut sys = GalacticDiskSystem::from_morphology(
            DwarfMorphology::TypeCode10,
            Mpc(10.0),
            0.5,
            0.6,
            HaloProfile::BaryonicOnly,
            OrbitPopulationConfig {
                disk_tracers: 4,
                tidal_tracers: 2,
                ..OrbitPopulationConfig::default()
            },
        );
        sys.step_n_leapfrog(2.0 / sub as f64, sub);
        let p = &sys.particles()[0];
        [p.x_kpc, p.y_kpc]
    };
    let ref_gal = eval_gal(512);
    let hs_gal = [2.0 / 16.0, 2.0 / 32.0, 2.0 / 64.0];
    let err_gal = [16usize, 32, 64].map(|s| {
        let p = eval_gal(s);
        (p[0] - ref_gal[0]).hypot(p[1] - ref_gal[1])
    });
    vec![
        phys_cert("FigureEightThreeBody", 3.85, &hs_nb, &err_nb, 1e-6),
        phys_cert("GalacticDiskSystem", 1.90, &hs_gal, &err_gal, 1e-4),
        certify_schwarzschild_bundle(),
    ]
}

fn certify_schwarzschild_bundle() -> SolverCertificate {
    let eval_gr = |n: usize| -> (nalgebra::Vector4<f64>, [f64; 2]) {
        let mut b = SchwarzschildGeodesicBundle::new(1.0, 2, 8).expect("valid r_s");
        b.seed_timelike_isco_comparison(4.0, 1.08, 0.05);
        for _ in 0..n {
            b.step_once(4.0 / n as f64);
        }
        let nw = b.newtonian_state();
        (b.trajectories()[0].pos, [nw.x, nw.y])
    };
    let (ref_gr, ref_nw) = eval_gr(320);
    let hs_gr = [4.0 / 8.0, 4.0 / 16.0, 4.0 / 32.0];
    let err_gr = [8usize, 16, 32].map(|n| {
        let (p, nw) = eval_gr(n);
        (p - ref_gr)
            .norm()
            .max((nw[0] - ref_nw[0]).hypot(nw[1] - ref_nw[1]))
    });
    phys_cert("SchwarzschildGeodesicBundle", 3.85, &hs_gr, &err_gr, 1e-6)
}

fn certify_chaos_solvers() -> Vec<SolverCertificate> {
    let eval_lorenz = |n: usize| -> Vector3<f64> {
        let mut sys = LorenzSystem::default_chaotic(LorenzState::new(1.0, 1.0, 1.0));
        for _ in 0..n {
            sys.step(0.08 / n as f64);
        }
        *sys.get_state()
    };
    let ref_lor = eval_lorenz(640);
    let hs_lor = [0.08 / 10.0, 0.08 / 20.0, 0.08 / 40.0];
    let err_lor = [10usize, 20, 40].map(|n| (eval_lorenz(n) - ref_lor).norm());

    let dp = DoublePendulum::new(1.0, 1.0, 1.0, 1.0, 9.81);
    let dp0 = ArrayState([0.3, -0.2, 0.0, 0.0]);
    let (mut dp_ref, mut rk_dp_ref) = (dp0, RungeKutta4::new(&dp0));
    for i in 0..640usize {
        let dt = 0.1 / 640.0;
        rk_dp_ref.step(&dp, i as f64 * dt, &mut dp_ref, dt);
    }
    let hs_dp = [0.1 / 10.0, 0.1 / 20.0, 0.1 / 40.0];
    let err_dp = [10usize, 20, 40].map(|n| {
        let dt = 0.1 / n as f64;
        let (mut s, mut rk) = (dp0, RungeKutta4::new(&dp0));
        for i in 0..n {
            rk.step(&dp, i as f64 * dt, &mut s, dt);
        }
        s.max_abs_diff(&dp_ref)
    });

    let hh = HenonHeiles::new(1.0);
    let hh0 = ArrayState([0.1, -0.1, 0.15, 0.1]);
    let (mut hh_ref, mut yo_hh_ref) = (hh0, Yoshida4::new(&ArrayState([0.0; 2])));
    for _ in 0..640 {
        yo_hh_ref.step_separable(&hh, &mut hh_ref, 0.2 / 640.0);
    }
    let hs_hh = [0.2 / 10.0, 0.2 / 20.0, 0.2 / 40.0];
    let err_hh = [10usize, 20, 40].map(|n| {
        let (mut s, mut yo) = (hh0, Yoshida4::new(&ArrayState([0.0; 2])));
        for _ in 0..n {
            yo.step_separable(&hh, &mut s, 0.2 / n as f64);
        }
        s.max_abs_diff(&hh_ref)
    });
    vec![
        phys_cert("LorenzSystem", 3.90, &hs_lor, &err_lor, 1e-5),
        phys_cert("DoublePendulum", 3.90, &hs_dp, &err_dp, 1e-6),
        phys_cert("HenonHeiles", 3.85, &hs_hh, &err_hh, 1e-6),
    ]
}

fn certify_optics_solvers() -> Vec<SolverCertificate> {
    let opt = OpticalMediumConfig {
        preset: OpticsPreset::MaxwellFishEye,
        ..OpticalMediumConfig::default()
    };
    let eval_ham = |n: usize| -> [f64; 2] {
        let mut tr = HamiltonianRayTracer::new(2, n, 16);
        tr.trace_in_place(&opt, 2, 0.2 / n as f64);
        let pts = tr.ray_points(0);
        pts[pts.len() - 1]
    };
    let ref_ham = eval_ham(640);
    let hs_ham = [0.2 / 10.0, 0.2 / 20.0, 0.2 / 40.0];
    let err_ham = [10usize, 20, 40].map(|n| {
        let p = eval_ham(n);
        (p[0] - ref_ham[0]).hypot(p[1] - ref_ham[1])
    });

    let (g, y0, tau_f) = (0.5_f64, 0.2_f64, 0.4_f64);
    let px0 = (1.0 - g * g * y0 * y0).sqrt();
    let eik = EikonalRaySystem {
        medium: GrinMedium::ParabolicWaveguide {
            n0: 1.0,
            gradient_g: g,
        },
    };
    let s0 = ArrayState([0.0, y0, px0, 0.0]);
    let exact = ArrayState([
        px0 * tau_f,
        y0 * (g * tau_f).cos(),
        px0,
        -g * y0 * (g * tau_f).sin(),
    ]);
    let hs_eik = [tau_f / 10.0, tau_f / 20.0, tau_f / 40.0];
    let err_eik = [10usize, 20, 40].map(|n| {
        let dt = tau_f / n as f64;
        let (mut s, mut rk) = (s0, RungeKutta4::new(&s0));
        for i in 0..n {
            rk.step(&eik, i as f64 * dt, &mut s, dt);
        }
        s.max_abs_diff(&exact)
    });
    vec![
        phys_cert("HamiltonianRayTracer", 3.85, &hs_ham, &err_ham, 1e-6),
        phys_cert("EikonalRaySystem", 3.90, &hs_eik, &err_eik, 1e-7),
    ]
}

fn certify_fluid_solvers() -> Result<Vec<SolverCertificate>, String> {
    let props = FluidProperties::new(2.0, 0.0).map_err(|e| format!("{e:?}"))?;
    let grads = SpatialGradients::new(
        Matrix3::zeros(),
        Vector3::new(1.0, -0.5, 0.0),
        Vector3::zeros(),
    );
    let mut fp = FluidParticleSystem::new(
        &props,
        EulerMomentum,
        grads,
        Vector3::new(0.2, 0.4, -0.1),
        FlowState::new(Vector3::new(1.0, 0.0, 0.5), 100.0),
    );
    fp.step(0.5);
    let err_fp = (fp.get_state().velocity - Vector3::new(0.85, 0.325, 0.45)).norm();

    let (tau, nu) = (0.8_f64, (0.8_f64 - 0.5) / 3.0);
    let _ = D2Q9::weights();
    let hs_lbm = [1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0];
    let mut err_lbm = [0.0; 3];
    for (idx, &nx) in [16usize, 32, 64].iter().enumerate() {
        let mut lbm: LatticeBoltzmannD2Q9<BgkCollision> = LatticeBoltzmannD2Q9::new(nx, 16, tau);
        let (eps, kw) = (0.02_f64, 2.0 * PI / nx as f64);
        for y in 0..16 {
            for x in 0..nx {
                lbm.set_inlet(x, y, 1, 1, 0.0, eps * (kw * x as f64).sin());
            }
        }
        for _ in 0..4 {
            lbm.step();
        }
        let exact_uy = eps * (-nu * kw * kw * 4.0).exp() * (kw * (nx / 4) as f64).sin();
        err_lbm[idx] = ((lbm.get_velocity(nx / 4, 8).1 - exact_uy) / eps).abs();
    }
    Ok(vec![
        SolverCertificate::evaluate(
            "FluidParticleSystem",
            "domain_physics",
            0.0,
            4.0,
            err_fp,
            1e-12,
        ),
        phys_cert("LatticeBoltzmann", 1.50, &hs_lbm, &err_lbm, 5e-2),
    ])
}

fn certify_mri_and_quantum() -> Vec<SolverCertificate> {
    let make_bloch = || {
        let mut bloch = BlochSimulator::new(Vector3::new(1.0, 0.0, 0.2), 1.0);
        bloch.set_b_field(Vector3::new(0.0, 0.0, 1e-7));
        bloch.set_relaxation(0.8, 0.4);
        bloch
    };
    let mut bloch_ref = make_bloch();
    for _ in 0..400usize {
        bloch_ref.step(0.02 / 400.0);
    }
    let hs_bl = [0.02 / 5.0, 0.02 / 10.0, 0.02 / 20.0];
    let err_bl = [5usize, 10, 20].map(|n| {
        let mut b = make_bloch();
        for _ in 0..n {
            b.step(0.02 / n as f64);
        }
        (b.get_state() - bloch_ref.get_state()).norm()
    });

    let n_q = 20usize;
    let pot = DVector::from_fn(n_q, |i, _| 0.3 * (-1.0 + 0.1 * i as f64).powi(2));
    let tri_h = TridiagonalHamiltonian1D::from_potential(&pot, 0.1, 1.0, 1.0);
    let mut dense_h = DMatrix::<Complex<f64>>::zeros(n_q, n_q);
    for i in 0..n_q {
        dense_h[(i, i)] = Complex::new(tri_h.diagonal()[i], 0.0);
        if i + 1 < n_q {
            dense_h[(i, i + 1)] = Complex::new(tri_h.off_diagonal()[i], 0.0);
            dense_h[(i + 1, i)] = Complex::new(tri_h.off_diagonal()[i], 0.0);
        }
    }
    let psi0 = QuantumState::new(DVector::from_fn(n_q, |i, _| {
        Complex::new((-0.2 * (i as f64 - 9.5).powi(2)).exp(), 0.0)
    }));
    let psi_exact =
        &time_evolution_operator(&QuantumOperator::new(dense_h), 0.01, 1.0).matrix * &psi0.vector;
    let hs_cn = [0.01 / 8.0, 0.01 / 16.0, 0.01 / 32.0];
    let err_cn = [8usize, 16, 32].map(|sub| {
        let mut cn = CrankNicolsonPropagator1D::new(&tri_h, 0.01 / sub as f64, 1.0);
        let mut st = psi0.clone();
        cn.evolve_substeps_in_place(&mut st, sub);
        (&st.vector - &psi_exact).norm()
    });
    vec![
        phys_cert("BlochSimulator", 3.85, &hs_bl, &err_bl, 1e-7),
        phys_cert("CrankNicolsonPropagator1D", 1.95, &hs_cn, &err_cn, 5e-3),
    ]
}

fn certify_bcs() -> Result<SolverCertificate, String> {
    let energies: Vec<f64> = (-20..=20).map(|k| 0.002 * k as f64).collect();
    let model = IsotropicBCSModel::new(energies, 0.018, 0.05).map_err(|e| format!("{e:?}"))?;
    let g_ref = BcsGapSolver::new()
        .with_max_iterations(500)
        .with_tolerance(1e-14)
        .solve(&model, ElectronVolts::new(0.02))
        .map_err(|e| format!("{e:?}"))?;
    let tols = [1e-4_f64, 1e-6, 1e-8];
    let err_bcs = tols.map(|tol| {
        let g = BcsGapSolver::new()
            .with_max_iterations(500)
            .with_tolerance(tol)
            .solve(&model, ElectronVolts::new(0.02))
            .unwrap();
        ((g - g_ref).abs() / ElectronVolts::new(1.0)).max(1e-16)
    });
    Ok(phys_cert("BcsGapSolver", 0.85, &tols, &err_bcs, 1e-7))
}

/// Certifies all 13 astrophysics, GR, chaos, optics, fluid, MRI, quantum, and solid-state solvers in `domain_physics`.
pub fn certify_physics_solvers() -> Result<Vec<SolverCertificate>, String> {
    let mut out = certify_astrophysics();
    out.extend(certify_chaos_solvers());
    out.extend(certify_optics_solvers());
    out.extend(certify_fluid_solvers()?);
    out.extend(certify_mri_and_quantum());
    out.push(certify_bcs()?);
    Ok(out)
}
