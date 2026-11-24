//! VerifiedAllocator lock tests confirming zero dynamic heap allocations across
//! 1,000 hot-loop steps of Nuclear, Standard Model, and High-Energy Observer kernels.

use domain_physics::physics::high_energy::{
    C, FourVector, apparent_transverse_beta, calculate_lorentz_factor, doppler_beaming_factor,
    fluid_dynamics::{shock_compression_ratio, specific_enthalpy},
    radiation::{inverse_compton_spectral_index, synchrotron_power},
    relativistic_doppler_factor,
    statistics::li_ma_significance,
};
use domain_physics::physics::nuclear::{
    AtomicNumber, BatemanChain, LiquidDropModel, MassNumber, bateman_two_step,
    decay::{calculate_remaining, gamow_factor},
    models::shell::spin_orbit_coupling,
    properties::{calculate_nucleon_density, calculate_radius},
    reactions::{breit_wigner, q_value},
};
use domain_physics::physics::standard_model::{
    GaugeCouplings, HiggsPotential, NeutrinoFlavor, PmnsParameters, boson_masses, construct_pmns,
    flavor::{construct_ckm, jarlskog_invariant, unitarity_triangle_apex},
    matter_potential_ev2, msw_oscillation_probabilities, pmns_oscillation_probabilities,
    qcd::running_coupling,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn step_nuclear_zero_alloc(ldm: &LiquidDropModel, step: usize) -> f64 {
    let z = AtomicNumber::new(26 + (step % 20) as u32);
    let a = MassNumber::new_unchecked(56 + (step % 45) as u32);
    let bd = ldm.binding_energy_breakdown(z, a).unwrap();
    let r = calculate_radius(a);
    let rho = calculate_nucleon_density(a).unwrap();
    let ls = spin_orbit_coupling(2.0, 0.5, 2.5);
    let eta = gamow_factor(z, AtomicNumber::new(2), 1.5e22).unwrap();
    let rem = calculate_remaining(100.0, 12.0, 6.0).unwrap();
    let (n1, n2) = bateman_two_step(100.0, 10.0, Some(10.0), 5.0).unwrap();
    let chain = BatemanChain::<4>::from_half_lives(
        [Some(12.0), Some(12.0), Some(6.0), None],
        [100.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let pops_short = chain.populations_at(0.4).unwrap();
    let pops_long = chain.populations_at(35.0).unwrap();
    let acts = chain.activities_at(15.0).unwrap();
    let qv = q_value(&[238.05], &[234.04, 4.0026]);
    let bw = breit_wigner(8.2, 8.0, 1.2).unwrap();

    assert!(BatemanChain::<4>::from_half_lives([Some(-1.0); 4], [1.0; 4]).is_err());
    assert!(chain.populations_at(-1.0).is_err());

    bd.total + r + rho + ls + eta + rem + n1 + n2 + pops_short[2] + pops_long[3] + acts[1] + qv + bw
}

fn step_standard_model_zero_alloc(pmns: &PmnsParameters, step: usize) -> f64 {
    let delta = 1.0 + 0.001 * (step as f64);
    let ckm = construct_ckm(0.227, 0.041, 0.0036, delta);
    let j = jarlskog_invariant(&ckm);
    let (rho_b, eta_b) = unitarity_triangle_apex(&ckm).unwrap_or((0.0, 0.0));
    let couplings = GaugeCouplings::new(0.357, 0.652, 1.221);
    let (cos_w, sin_w) = couplings.weak_mixing_angle();
    let higgs = HiggsPotential {
        mu2: -7840.0,
        lambda: 0.1293,
    };
    let vev = higgs.vev().unwrap();
    let (mw, mz) = boson_masses(vev, &couplings);
    let alpha_s = running_coupling(91.1876, 0.1179, 20.0 + (step % 50) as f64, 5.0).unwrap();

    let u_pmns = construct_pmns(pmns);
    let l_km = 295.0 + (step % 500) as f64;
    let acc = matter_potential_ev2(2.5, 2.8, 0.5, false).unwrap();
    let vac = pmns_oscillation_probabilities(pmns, NeutrinoFlavor::Muon, l_km, 2.5, false).unwrap();
    let msw = msw_oscillation_probabilities(pmns, NeutrinoFlavor::Muon, l_km, 2.5, 2.8, 0.5, false)
        .unwrap();

    assert!(matter_potential_ev2(0.0, 2.8, 0.5, false).is_err());
    j + rho_b
        + eta_b
        + cos_w
        + sin_w
        + mw
        + mz
        + alpha_s
        + u_pmns[(0, 0)].re
        + acc
        + vac[0]
        + msw[0]
}

fn step_high_energy_zero_alloc(step: usize) -> f64 {
    let beta = 0.5 + 0.0004 * ((step % 1000) as f64);
    let gamma = calculate_lorentz_factor(beta * C).unwrap();
    let rest_p = FourVector::new(2.0 * C, 0.3 * C, 0.0, 0.0);
    let p_norm = rest_p.minkowski_norm_squared();
    let boosted = rest_p.lorentz_boost(-beta, 0.0, 0.0).unwrap();
    let evt = FourVector::new(1.0 / C, 2.0, 0.0, 0.0)
        .lorentz_boost_spacetime(0.3 * C, 0.0, 0.0)
        .unwrap();
    let doppler = relativistic_doppler_factor(beta, 0.15).unwrap();
    let beta_app = apparent_transverse_beta(beta, 0.15).unwrap();
    let beam = doppler_beaming_factor(doppler, 0.7, true).unwrap();
    let psync = synchrotron_power(1.5, beta, gamma).unwrap();
    let ic = inverse_compton_spectral_index(2.4).unwrap();
    let shock = shock_compression_ratio(4.0 / 3.0).unwrap();
    let enth = specific_enthalpy(4.0 / 3.0, 0.2 * C * C, 1.0).unwrap();
    let lima = li_ma_significance(180.0, 200.0, 0.25).unwrap();

    assert!(rest_p.lorentz_boost(0.9, 0.9, 0.0).is_err());
    assert!(relativistic_doppler_factor(beta, f64::NAN).is_err());
    assert!(doppler_beaming_factor(-1.0, 0.5, true).is_err());

    p_norm / (C * C) + boosted.t / C + evt.x + beam + psync + ic + shock + enth + beta_app + lima
}

#[test]
fn test_1000_subatomic_and_relativistic_steps_zero_heap_allocations() {
    let ldm = LiquidDropModel::new();
    let pmns = PmnsParameters::pdg_normal_ordering();

    let (checksum, _metrics) = VerifiedEngine::run_verified(|| {
        let mut acc = 0.0_f64;
        for step in 0..1_000 {
            acc += step_nuclear_zero_alloc(&ldm, step);
            acc += step_standard_model_zero_alloc(&pmns, step);
            acc += step_high_energy_zero_alloc(step);
        }
        acc
    });

    assert!(checksum.is_finite());
}
