//! Battery degradation models.
// [cite:battery_degradation]

use super::types::{Capacity, Cycles, DepthOfDischarge};
use verified_engine::Theory;

/// Minimum supported cell operating temperature in degrees Celsius (`253.15 K`).
pub const MIN_TEMPERATURE_C: f64 = -20.0;
/// Maximum supported cell operating temperature in degrees Celsius (`333.15 K`).
pub const MAX_TEMPERATURE_C: f64 = 60.0;
/// Offset converting degrees Celsius to Kelvin.
pub const CELSIUS_TO_KELVIN: f64 = 273.15;
/// Standard electrochemical reference temperature in Kelvin (`25.0 °C`).
pub const REFERENCE_TEMPERATURE_K: f64 = 298.15;
/// Universal molar gas constant $R$ in $\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$.
pub const GAS_CONSTANT_R: f64 = 8.314_462_618;
/// Minimum supported charge/discharge C-rate (`1/h`).
pub const MIN_C_RATE: f64 = 0.05;
/// Maximum supported charge/discharge C-rate (`1/h`).
pub const MAX_C_RATE: f64 = 10.0;
/// Minimum supported stretched cycling exponent $z$.
pub const MIN_EXPONENT_Z: f64 = 0.1;
/// Maximum supported stretched cycling exponent $z$.
pub const MAX_EXPONENT_Z: f64 = 2.0;

#[allow(missing_docs)]
pub trait DegradationModel {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn n70(&self, d: DepthOfDischarge) -> Cycles;

    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity;

    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles;
}

#[derive(Debug, Clone, Copy, Theory)]
#[theory(
    description = "Empirical power-law model characterizing the cycle life of lithium-ion batteries as a function of depth of discharge.",
    citation = "Capacity Fade Model for Li-ion Batteries (Smith et al., 2011)"
)]
#[allow(missing_docs)]
pub struct PowerLawModel {
    alpha: f64,
    beta: f64,
}

impl PowerLawModel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn standard() -> Self {
        Self {
            alpha: 1.019e5,
            beta: -1.2639,
        }
    }
}

impl DegradationModel for PowerLawModel {
    #[verified_engine::verified]
    fn n70(&self, d: DepthOfDischarge) -> Cycles {
        let raw_d = d / DepthOfDischarge::new_clamped(1.0);
        let val = self.alpha * raw_d.powf(self.beta);
        Cycles::new_clamped(val)
    }

    #[verified_engine::verified]
    fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity {
        let n70_val = self.n70(d) / Cycles::new_clamped(1.0);
        if n70_val == 0.0 {
            return Capacity::new_clamped(0.0);
        }
        let n_val = n / Cycles::new_clamped(1.0);
        let cap = 0.7_f64.powf(n_val / n70_val);
        Capacity::new_clamped(cap)
    }

    #[verified_engine::verified]
    fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles {
        let n70_val = self.n70(d) / Cycles::new_clamped(1.0);
        let target_raw = target / Capacity::new_clamped(1.0);
        let val = (target_raw.ln() / math_commons::constants::LN_0_7) * n70_val;
        Cycles::new_clamped(val)
    }
}

impl PowerLawModel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn n70(&self, d: DepthOfDischarge) -> Cycles {
        DegradationModel::n70(self, d)
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity {
        DegradationModel::capacity(self, n, d)
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles {
        DegradationModel::cycles_to_capacity(self, target, d)
    }
}

mod preset;
pub use preset::ChemistryPreset;

fn sanitize_bounded(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

/// Coupled electrochemical aging model combining DoD stress, stretched exponent $z$,
/// Arrhenius thermal activation, C-rate mechanical stress, and $\sqrt{t}$ SEI calendar fade.
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[theory(
    description = "Coupled electrochemical Li-ion aging model combining stretched power-law cycling fade 0.3*(N/N_70)^z, Arrhenius thermal activation exp(-E_a/(R*T)), C-rate stress, and parabolic sqrt(t) SEI growth.",
    citation = "Cycle-life model for graphite-LiFePO4 cells (Wang et al., 2011); Capacity Fade Model for Li-ion Batteries (Smith et al., 2011)"
)]
pub struct ElectrochemicalAgingModel {
    preset: ChemistryPreset,
    alpha: f64,
    beta: f64,
    z: f64,
    activation_energy: f64,
    c_rate_exponent: f64,
    sei_rate: f64,
    temperature_c: f64,
    c_rate: f64,
}

impl Default for ElectrochemicalAgingModel {
    fn default() -> Self {
        Self::from_preset(ChemistryPreset::Nmc811)
    }
}

impl ElectrochemicalAgingModel {
    /// Initializes a coupled aging model from [`ChemistryPreset`] at $25\,^\circ\text{C}$ and $1.0\text{C}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_preset(preset: ChemistryPreset) -> Self {
        Self {
            preset,
            alpha: preset.default_alpha(),
            beta: preset.default_beta(),
            z: preset.default_exponent_z(),
            activation_energy: preset.default_activation_energy(),
            c_rate_exponent: preset.default_c_rate_exponent(),
            sei_rate: preset.default_sei_rate(),
            temperature_c: 25.0,
            c_rate: 1.0,
        }
    }

    /// Constructs a coupled aging model from `preset` kinetic defaults with custom DoD/operating parameters.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(
        preset: ChemistryPreset,
        alpha: f64,
        beta: f64,
        z: f64,
        temperature_c: f64,
        c_rate: f64,
    ) -> Self {
        Self::from_preset(preset)
            .with_dod_law(alpha, beta)
            .with_exponent_z(z)
            .with_temperature_c(temperature_c)
            .with_c_rate(c_rate)
    }

    /// Returns a copy with custom kinetic activation energy, C-rate exponent, and SEI calendar rate.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_kinetics(
        mut self,
        activation_energy: f64,
        c_rate_exponent: f64,
        sei_rate: f64,
    ) -> Self {
        self.activation_energy = sanitize_bounded(
            activation_energy,
            0.0,
            120_000.0,
            self.preset.default_activation_energy(),
        );
        self.c_rate_exponent = sanitize_bounded(
            c_rate_exponent,
            0.0,
            3.0,
            self.preset.default_c_rate_exponent(),
        );
        self.sei_rate = sanitize_bounded(sei_rate, 0.0, 0.2, self.preset.default_sei_rate());
        self
    }

    /// Returns a copy with operating temperature clamped to `[-20.0, 60.0]` °C (`253.15..=333.15` K).
    #[must_use]
    #[verified_engine::verified]
    pub fn with_temperature_c(mut self, temperature_c: f64) -> Self {
        self.temperature_c =
            sanitize_bounded(temperature_c, MIN_TEMPERATURE_C, MAX_TEMPERATURE_C, 25.0);
        self
    }

    /// Returns a copy with operating C-rate clamped to `[0.05, 10.0]` `1/h`.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_c_rate(mut self, c_rate: f64) -> Self {
        self.c_rate = sanitize_bounded(c_rate, MIN_C_RATE, MAX_C_RATE, 1.0);
        self
    }

    /// Returns a copy with stretched cycling exponent $z$ clamped to `[0.1, 2.0]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_exponent_z(mut self, z: f64) -> Self {
        self.z = sanitize_bounded(z, MIN_EXPONENT_Z, MAX_EXPONENT_Z, self.z);
        self
    }

    /// Returns a copy with parabolic SEI calendar fade rate $k_{\text{SEI}}$ (`day^-0.5` at `25 °C`).
    #[must_use]
    #[verified_engine::verified]
    pub fn with_sei_rate(mut self, sei_rate: f64) -> Self {
        self.sei_rate = sanitize_bounded(sei_rate, 0.0, 0.2, self.sei_rate);
        self
    }

    /// Returns a copy with updated DoD power-law prefactor $\alpha$ and exponent $\beta$.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_dod_law(mut self, alpha: f64, beta: f64) -> Self {
        self.alpha = sanitize_bounded(alpha, 1.0, 1.0e9, self.alpha);
        self.beta = sanitize_bounded(beta, -4.0, -0.05, self.beta);
        self
    }

    /// Returns the base [`ChemistryPreset`] associated with this model.
    #[must_use]
    #[verified_engine::verified]
    pub const fn preset(&self) -> ChemistryPreset {
        self.preset
    }

    /// Returns the DoD power-law prefactor $\alpha$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the DoD power-law exponent $\beta$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Returns the stretched cycling exponent $z \in [0.1, 2.0]$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn exponent_z(&self) -> f64 {
        self.z
    }

    /// Returns the Arrhenius activation energy $E_a$ ($\text{J}\cdot\text{mol}^{-1}$).
    #[must_use]
    #[verified_engine::verified]
    pub const fn activation_energy(&self) -> f64 {
        self.activation_energy
    }

    /// Returns the C-rate stress exponent $\gamma_c$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn c_rate_exponent(&self) -> f64 {
        self.c_rate_exponent
    }

    /// Returns the reference $25\,^\circ\text{C}$ SEI calendar fade rate ($\text{day}^{-1/2}$).
    #[must_use]
    #[verified_engine::verified]
    pub const fn sei_rate(&self) -> f64 {
        self.sei_rate
    }

    /// Returns the clamped operating temperature in degrees Celsius (`-20.0..=60.0`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn temperature_c(&self) -> f64 {
        self.temperature_c
    }

    /// Returns the clamped absolute operating temperature in Kelvin (`253.15..=333.15`).
    #[must_use]
    #[verified_engine::verified]
    pub fn temperature_k(&self) -> f64 {
        self.temperature_c + CELSIUS_TO_KELVIN
    }

    /// Returns the clamped operating C-rate (`0.05..=10.0`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn c_rate(&self) -> f64 {
        self.c_rate
    }

    /// Normalized Arrhenius thermal acceleration factor $\exp((E_a / R)(1/T_{\text{ref}} - 1/T))$, equal to `1.0` at $25\,^\circ\text{C}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn arrhenius_factor(&self) -> f64 {
        let exponent = (self.activation_energy / GAS_CONSTANT_R)
            * (1.0 / REFERENCE_TEMPERATURE_K - 1.0 / self.temperature_k());
        exponent.exp()
    }

    /// Dimensionless C-rate mechanical stress factor $I_c^{\gamma_c}$ (`1.0` at $1\text{C}$).
    #[must_use]
    #[verified_engine::verified]
    pub fn c_rate_stress_factor(&self) -> f64 {
        self.c_rate.powf(self.c_rate_exponent)
    }

    /// Fractional cycling capacity loss $0.3 \cdot (N / N_{70}(D, T, I_c))^z$.
    #[must_use]
    #[verified_engine::verified]
    pub fn cycling_loss(&self, n: Cycles, d: DepthOfDischarge) -> f64 {
        let raw_d = d.as_f64();
        let raw_n = n.as_f64();
        if raw_n <= 0.0 || raw_d <= 0.0 {
            return 0.0;
        }
        let n70_val = self.n70(d).as_f64();
        if n70_val <= 0.0 {
            return 1.0;
        }
        (0.3 * (raw_n / n70_val).powf(self.z)).clamp(0.0, 1.0)
    }

    /// Fractional SEI calendar capacity loss $k_{\text{SEI}} A_T(T) \sqrt{t_{\text{days}}}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn sei_calendar_loss(&self, elapsed_days: f64) -> f64 {
        if !elapsed_days.is_finite() || elapsed_days <= 0.0 {
            return 0.0;
        }
        (self.sei_rate * self.arrhenius_factor() * elapsed_days.sqrt()).clamp(0.0, 1.0)
    }

    /// Coupled state-of-health capacity combining stretched cycling fade and $\sqrt{t}$ SEI growth.
    #[must_use]
    #[verified_engine::verified]
    pub fn coupled_capacity(&self, n: Cycles, d: DepthOfDischarge, elapsed_days: f64) -> Capacity {
        let loss = self.cycling_loss(n, d) + self.sei_calendar_loss(elapsed_days);
        Capacity::new_clamped(1.0 - loss)
    }

    /// Projected cycle life to $70\%$ capacity under pure cycling at `(d, temperature_c, c_rate)`.
    #[must_use]
    #[verified_engine::verified]
    pub fn n70(&self, d: DepthOfDischarge) -> Cycles {
        DegradationModel::n70(self, d)
    }

    /// Cycling state-of-health capacity after `n` cycles at depth of discharge `d`.
    #[must_use]
    #[verified_engine::verified]
    pub fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity {
        DegradationModel::capacity(self, n, d)
    }

    /// Cycles required to reach `target` state of health under cycling at depth of discharge `d`.
    #[must_use]
    #[verified_engine::verified]
    pub fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles {
        DegradationModel::cycles_to_capacity(self, target, d)
    }
}

impl DegradationModel for ElectrochemicalAgingModel {
    #[verified_engine::verified]
    fn n70(&self, d: DepthOfDischarge) -> Cycles {
        let raw_d = d.as_f64().clamp(1.0e-4, 100.0);
        let stress = self.arrhenius_factor() * self.c_rate_stress_factor();
        let val = (self.alpha * raw_d.powf(self.beta)) / stress.max(1.0e-12);
        Cycles::new_clamped(val)
    }

    #[verified_engine::verified]
    fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity {
        Capacity::new_clamped(1.0 - self.cycling_loss(n, d))
    }

    #[verified_engine::verified]
    fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles {
        let loss = (1.0 - target.as_f64()).clamp(0.0, 1.0);
        if loss <= 0.0 {
            return Cycles::new_clamped(0.0);
        }
        let n70_val = DegradationModel::n70(self, d).as_f64();
        let ratio = (loss / 0.3).powf(1.0 / self.z);
        Cycles::new_clamped(n70_val * ratio)
    }
}

#[cfg(test)]
mod tests;
