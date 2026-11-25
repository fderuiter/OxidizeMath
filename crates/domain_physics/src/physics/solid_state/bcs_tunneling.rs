//! Dynes quasiparticle density of states and finite-temperature NIS tunneling spectroscopy.

use super::super::types::{ElectronVolts, Kelvin};
use crate::error::SolidStateError;
use num_complex::Complex;

#[inline]
fn dynes_dos_raw(energy_ev: f64, delta_ev: f64, gamma_ev: f64) -> f64 {
    if delta_ev <= 1e-15 {
        return 1.0;
    }
    let e_abs = energy_ev.abs();
    if gamma_ev <= 1e-15 {
        return if e_abs <= delta_ev {
            0.0
        } else {
            e_abs / (e_abs * e_abs - delta_ev * delta_ev).sqrt()
        };
    }
    let z = Complex::new(e_abs, gamma_ev);
    let root = (z * z - Complex::new(delta_ev * delta_ev, 0.0)).sqrt();
    (z / root).re.max(0.0)
}

/// Evaluates normalized Dynes quasiparticle density of states $N_S(E) / N(0) = \operatorname{Re}\!\left[\frac{|E| + i\Gamma}{\sqrt{(|E| + i\Gamma)^2 - \Delta^2}}\right]$.
///
/// # Errors
///
/// Returns [`SolidStateError::InvalidParameter`] if `gamma < 0` or any input is non-finite.
#[verified_engine::verified]
pub fn dynes_dos_ratio(
    energy: ElectronVolts,
    delta: ElectronVolts,
    gamma: ElectronVolts,
) -> Result<f64, SolidStateError> {
    let e = energy.as_f64();
    let d = delta.as_f64().abs();
    let g = gamma.as_f64();
    if !e.is_finite() || !d.is_finite() || !g.is_finite() || g < 0.0 {
        return Err(SolidStateError::InvalidParameter(
            "Dynes broadening gamma must be finite and non-negative".to_string(),
        ));
    }
    Ok(dynes_dos_raw(e, d, g))
}

// 16-point positive half-interval Gauss-Legendre abscissas and weights on [0, 1] (32 symmetric nodes on [-1, 1]).
const GL32_POS_NODES: [f64; 16] = [
    0.048_307_665_687_738_32,
    0.144_471_961_582_796_5,
    0.239_287_362_252_137_1,
    0.331_868_602_282_127_67,
    0.421_351_276_130_635_33,
    0.506_899_908_932_229_4,
    0.587_715_757_240_762_3,
    0.663_044_266_930_215_2,
    0.732_182_118_740_289_7,
    0.794_483_795_967_942_4,
    0.849_367_613_732_57,
    0.896_321_155_766_052_1,
    0.934_906_075_937_739_7,
    0.964_762_255_587_506_4,
    0.985_611_511_545_268_4,
    0.997_263_861_849_481_6,
];

const GL32_POS_WEIGHTS: [f64; 16] = [
    0.096_540_088_514_727_8,
    0.095_638_720_079_274_86,
    0.093_844_399_080_804_57,
    0.091_173_878_695_763_88,
    0.087_652_093_004_403_81,
    0.083_311_924_226_946_75,
    0.078_193_895_787_070_3,
    0.072_345_794_108_848_5,
    0.065_822_222_776_361_85,
    0.058_684_093_478_535_54,
    0.050_998_059_262_376_176,
    0.042_835_898_022_226_68,
    0.034_273_862_913_021_43,
    0.025_392_065_309_262_06,
    0.016_274_394_730_905_67,
    0.007_018_610_009_470_097,
];

/// Evaluates normalized finite-temperature NIS differential tunneling conductance $G_{NS}(V) / G_{NN}$ (`dI/dV`).
///
/// Convolves the Dynes quasiparticle density of states with the thermal Fermi-Dirac kernel
/// $-\partial f(E - eV) / \partial E$ via zero-allocation 32-point Gauss-Legendre quadrature on
/// $s = \tanh((E - eV)/(2 k_B T)) \in (-1, 1)$.
///
/// # Errors
///
/// Returns [`SolidStateError::InvalidParameter`] if `gamma < 0`, `temperature < 0`, or any parameter is non-finite.
#[verified_engine::verified]
pub fn nis_tunneling_conductance(
    bias: ElectronVolts,
    delta: ElectronVolts,
    gamma: ElectronVolts,
    temperature: Kelvin,
) -> Result<f64, SolidStateError> {
    let v_ev = bias.as_f64();
    let d_ev = delta.as_f64().abs();
    let g_ev = gamma.as_f64();
    let t_k = temperature.as_f64();
    if !v_ev.is_finite()
        || !d_ev.is_finite()
        || !g_ev.is_finite()
        || !t_k.is_finite()
        || g_ev < 0.0
        || t_k < 0.0
    {
        return Err(SolidStateError::InvalidParameter(
            "NIS tunneling bias, gap, gamma >= 0, and temperature >= 0 must be finite".to_string(),
        ));
    }
    if d_ev <= 1e-15 {
        return Ok(1.0);
    }
    let two_kt_ev = 2.0 * temperature.thermal_energy_ev().as_f64();
    if two_kt_ev <= 1e-9 {
        return Ok(dynes_dos_raw(v_ev, d_ev, g_ev));
    }
    let mut accum = 0.0;
    for i in 0..16 {
        let s = GL32_POS_NODES[i];
        let w = GL32_POS_WEIGHTS[i];
        let shift = two_kt_ev * s.atanh();
        let dos_plus = dynes_dos_raw(v_ev + shift, d_ev, g_ev);
        let dos_minus = dynes_dos_raw(v_ev - shift, d_ev, g_ev);
        accum += w * (dos_plus + dos_minus);
    }
    Ok(0.5 * accum)
}

// [cite:mmwave_radiotherapy_setup]
