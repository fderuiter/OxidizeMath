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

/// Trait characterizing Li-ion state-of-health capacity fade $C(N, D) \in [0, 1]$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`).
/// Numerical stability: all outputs clamp safely to bounded `Cycles`/`Capacity` domains. Domain of validity: $D \in (0, 1]$, $N \ge 0$, and $C \in (0, 1]$.
pub trait DegradationModel {
    /// Computes cycle life $N_{70}(D)$ at which $C(N_{70}, D) = 0.7$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: clamped non-negative `Cycles`. Domain of validity: $D \in (0, 1]$.
    #[verified_engine::verified]
    fn n70(&self, d: DepthOfDischarge) -> Cycles;

    /// Evaluates remaining fractional capacity $C(N, D) \in [0, 1]$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: guards $N_{70} = 0$ without zero division. Domain of validity: $N \ge 0, D \in (0, 1]$.
    #[verified_engine::verified]
    fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity;

    /// Inverts capacity fade for cycle count $N(C_{\text{target}}, D)$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: clamped log-ratio evaluation. Domain of validity: $C_{\text{target}} \in (0, 1], D \in (0, 1]$.
    #[verified_engine::verified]
    fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles;
}

/// Empirical DoD power-law cycle-life model $N_{70}(D) = \alpha D^\beta, C(N, D) = 0.7^{N / N_{70}(D)}$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`).
/// Numerical stability: `Cycles`/`Capacity` clamping prevents overflow near $D \to 0^+$. Domain of validity: $\alpha > 0, \beta < 0, D \in (0, 1]$.
#[derive(Debug, Clone, Copy, Theory)]
#[theory(
    description = "Empirical power-law model characterizing the cycle life of lithium-ion batteries as a function of depth of discharge.",
    citation = "bloom2001accelerated"
)]
pub struct PowerLawModel {
    alpha: f64,
    beta: f64,
}

impl PowerLawModel {
    /// Constructs $N_{70}(D) = \alpha D^\beta$ from $(\alpha, \beta)$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` capture. Domain of validity: finite $\alpha > 0, \beta < 0$.
    #[verified_engine::verified]
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Returns reference Li-ion fit $(\alpha, \beta) = (1.019 \times 10^5, -1.2639)$ for $N_{70}(D) = \alpha D^\beta$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: well-conditioned on $D \in [0.05, 1.0]$. Domain of validity: $D \in (0, 1]$ at $25\,^\circ\text{C}$.
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
    /// Evaluates $N_{70}(D) = \alpha D^\beta$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: clamped `Cycles` output. Domain of validity: $D \in (0, 1]$.
    #[verified_engine::verified]
    pub fn n70(&self, d: DepthOfDischarge) -> Cycles {
        DegradationModel::n70(self, d)
    }

    /// Evaluates $C(N, D) = 0.7^{N / N_{70}(D)}$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: guards $N_{70} = 0$ safely. Domain of validity: $N \ge 0, D \in (0, 1]$.
    #[verified_engine::verified]
    pub fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity {
        DegradationModel::capacity(self, n, d)
    }

    /// Evaluates $N(C, D) = N_{70}(D) \ln C / \ln 0.7$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: uses precomputed `LN_0_7` constant. Domain of validity: $C \in (0, 1], D \in (0, 1]$.
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

/// Coupled electrochemical aging model $C = 1 - 0.3(N/N_{70})^z - k_{\text{SEI}} A_T(T)\sqrt{t}$ (`@ning2004cycle`, `@bloom2001accelerated`, `spec:battery_degradation_v1`).
/// Numerical stability: clamps temperature, C-rate, and $z$ away from singular exponents. Domain of validity: $T \in [-20, 60]\,^\circ\text{C}, I_c \in [0.05, 10]\,\text{C}, z \in [0.1, 2.0]$.
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[theory(
    description = "Coupled electrochemical Li-ion aging model combining stretched power-law cycling fade 0.3*(N/N_70)^z, Arrhenius thermal activation exp(-E_a/(R*T)), C-rate stress, and parabolic sqrt(t) SEI growth.",
    citation = "ning2004cycle, bloom2001accelerated"
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
    /// Initializes coupled fade $C = 1 - Q_{\text{cyc}} - Q_{\text{SEI}}$ from [`ChemistryPreset`] at $T = 25\,^\circ\text{C}, I_c = 1\text{C}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: preset constants lie strictly inside bounded intervals. Domain of validity: all supported [`ChemistryPreset`] variants.
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

    /// Constructs $N_{70}(D, T, I_c) = \alpha D^\beta / (A_T I_c^{\gamma_c})$ with clamped parameters (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: `sanitize_bounded` replaces `NaN`/`Inf` with safe defaults. Domain of validity: clamped to $T \in [-20, 60]\,^\circ\text{C}, I_c \in [0.05, 10]\,\text{C}, z \in [0.1, 2.0]$.
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

    /// Updates kinetic parameters $(E_a, \gamma_c, k_{\text{SEI}})$ in $A_T(T) I_c^{\gamma_c}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: clamps $E_a \le 120\,\text{kJ/mol}$ to prevent `exp` overflow. Domain of validity: $E_a \in [0, 1.2 \times 10^5], \gamma_c \in [0, 3], k_{\text{SEI}} \in [0, 0.2]$.
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

    /// Sets operating temperature $T_C \in [-20, 60]\,^\circ\text{C}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: keeps $T_K \ge 253.15\,\text{K}$ away from $1/T_K$ singularity. Domain of validity: $T_C \in [-20.0, 60.0]\,^\circ\text{C}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_temperature_c(mut self, temperature_c: f64) -> Self {
        self.temperature_c =
            sanitize_bounded(temperature_c, MIN_TEMPERATURE_C, MAX_TEMPERATURE_C, 25.0);
        self
    }

    /// Sets operating C-rate $I_c \in [0.05, 10.0]$ in $I_c^{\gamma_c}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: positive lower bound prevents zero-stress division. Domain of validity: $I_c \in [0.05, 10.0]\,\text{h}^{-1}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_c_rate(mut self, c_rate: f64) -> Self {
        self.c_rate = sanitize_bounded(c_rate, MIN_C_RATE, MAX_C_RATE, 1.0);
        self
    }

    /// Sets stretched cycling exponent $z \in [0.1, 2.0]$ in $0.3(N/N_{70})^z$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: bounds $1/z \le 10$ during inverse cycle solves. Domain of validity: $z \in [0.1, 2.0]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_exponent_z(mut self, z: f64) -> Self {
        self.z = sanitize_bounded(z, MIN_EXPONENT_Z, MAX_EXPONENT_Z, self.z);
        self
    }

    /// Sets parabolic SEI rate $k_{\text{SEI}}$ in $Q_{\text{SEI}} = k_{\text{SEI}} A_T \sqrt{t}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: bounded in $[0, 0.2]$. Domain of validity: $k_{\text{SEI}} \in [0.0, 0.2]\,\text{day}^{-1/2}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_sei_rate(mut self, sei_rate: f64) -> Self {
        self.sei_rate = sanitize_bounded(sei_rate, 0.0, 0.2, self.sei_rate);
        self
    }

    /// Updates DoD law parameters $(\alpha, \beta)$ in $N_{70} \propto \alpha D^\beta$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: clamps $\alpha \in [1, 10^9], \beta \in [-4, -0.05]$. Domain of validity: finite $\alpha > 0, \beta < 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_dod_law(mut self, alpha: f64, beta: f64) -> Self {
        self.alpha = sanitize_bounded(alpha, 1.0, 1.0e9, self.alpha);
        self.beta = sanitize_bounded(beta, -4.0, -0.05, self.beta);
        self
    }

    /// Returns chemistry preset governing baseline $(E_a, \gamma_c, k_{\text{SEI}})$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: exact enum copy. Domain of validity: all [`ChemistryPreset`] variants.
    #[must_use]
    #[verified_engine::verified]
    pub const fn preset(&self) -> ChemistryPreset {
        self.preset
    }

    /// Returns DoD power-law prefactor $\alpha > 0$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` copy. Domain of validity: $\alpha \in [1, 10^9]$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns DoD power-law exponent $\beta < 0$ (`@bloom2001accelerated`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` copy. Domain of validity: $\beta \in [-4.0, -0.05]$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Returns stretched cycling exponent $z \in [0.1, 2.0]$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` copy. Domain of validity: $z \in [0.1, 2.0]$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn exponent_z(&self) -> f64 {
        self.z
    }

    /// Returns Arrhenius activation energy $E_a$ ($\text{J}\cdot\text{mol}^{-1}$) (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` copy. Domain of validity: $E_a \in [0, 1.2 \times 10^5]$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn activation_energy(&self) -> f64 {
        self.activation_energy
    }

    /// Returns C-rate mechanical stress exponent $\gamma_c$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` copy. Domain of validity: $\gamma_c \in [0.0, 3.0]$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn c_rate_exponent(&self) -> f64 {
        self.c_rate_exponent
    }

    /// Returns reference $25\,^\circ\text{C}$ SEI fade rate $k_{\text{SEI}}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` copy. Domain of validity: $k_{\text{SEI}} \in [0.0, 0.2]\,\text{day}^{-1/2}$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn sei_rate(&self) -> f64 {
        self.sei_rate
    }

    /// Returns clamped temperature $T_C \in [-20, 60]\,^\circ\text{C}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` copy. Domain of validity: $[-20.0, 60.0]\,^\circ\text{C}$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn temperature_c(&self) -> f64 {
        self.temperature_c
    }

    /// Returns absolute temperature $T_K = T_C + 273.15$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: single `f64` addition bounded in $[253.15, 333.15]$. Domain of validity: $T_C \in [-20, 60]\,^\circ\text{C}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn temperature_k(&self) -> f64 {
        self.temperature_c + CELSIUS_TO_KELVIN
    }

    /// Returns clamped operating C-rate $I_c \in [0.05, 10.0]$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: exact `f64` copy. Domain of validity: $[0.05, 10.0]\,\text{h}^{-1}$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn c_rate(&self) -> f64 {
        self.c_rate
    }

    /// Evaluates Arrhenius factor $A_T(T) = \exp((E_a/R)(T_{\text{ref}}^{-1} - T^{-1}))$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: exponent bounded on $T \in [253.15, 333.15]\,\text{K}$ without overflow. Domain of validity: $E_a \in [0, 1.2 \times 10^5], T \in [253.15, 333.15]\,\text{K}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn arrhenius_factor(&self) -> f64 {
        let exponent = (self.activation_energy / GAS_CONSTANT_R)
            * (1.0 / REFERENCE_TEMPERATURE_K - 1.0 / self.temperature_k());
        exponent.exp()
    }

    /// Evaluates C-rate stress factor $S_I = I_c^{\gamma_c}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: well-conditioned `powf` on $I_c \in [0.05, 10], \gamma_c \in [0, 3]$. Domain of validity: $I_c \in [0.05, 10.0], \gamma_c \in [0, 3]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn c_rate_stress_factor(&self) -> f64 {
        self.c_rate.powf(self.c_rate_exponent)
    }

    /// Evaluates fractional cycling loss $Q_{\text{cyc}} = 0.3(N / N_{70})^z \in [0, 1]$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: guards $N_{70} \le 0$ and clamps output to $[0, 1]$. Domain of validity: $N \ge 0, D \in (0, 1]$.
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

    /// Evaluates parabolic SEI loss $Q_{\text{SEI}} = k_{\text{SEI}} A_T(T)\sqrt{t} \in [0, 1]$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: returns `0.0` for non-positive/non-finite `elapsed_days` and clamps to $[0, 1]$. Domain of validity: finite $t \ge 0$ days.
    #[must_use]
    #[verified_engine::verified]
    pub fn sei_calendar_loss(&self, elapsed_days: f64) -> f64 {
        if !elapsed_days.is_finite() || elapsed_days <= 0.0 {
            return 0.0;
        }
        (self.sei_rate * self.arrhenius_factor() * elapsed_days.sqrt()).clamp(0.0, 1.0)
    }

    /// Evaluates coupled capacity $C(N, D, t) = \text{clamp}(1 - Q_{\text{cyc}} - Q_{\text{SEI}}, 0, 1)$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: bounded subtraction clamped to `[0, 1]`. Domain of validity: $N \ge 0, D \in (0, 1], t \ge 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn coupled_capacity(&self, n: Cycles, d: DepthOfDischarge, elapsed_days: f64) -> Capacity {
        let loss = self.cycling_loss(n, d) + self.sei_calendar_loss(elapsed_days);
        Capacity::new_clamped(1.0 - loss)
    }

    /// Evaluates stressed cycle life $N_{70}(D, T, I_c) = \alpha D^\beta / (A_T I_c^{\gamma_c})$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: clamps $D \ge 10^{-4}$ and stress $\ge 10^{-12}$. Domain of validity: $D \in (0, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn n70(&self, d: DepthOfDischarge) -> Cycles {
        DegradationModel::n70(self, d)
    }

    /// Evaluates pure-cycling state of health $C(N, D) = 1 - 0.3(N / N_{70})^z$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: clamped in `[0, 1]`. Domain of validity: $N \ge 0, D \in (0, 1]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity {
        DegradationModel::capacity(self, n, d)
    }

    /// Solves $N(C_{\text{target}}, D) = N_{70} ((1 - C_{\text{target}})/0.3)^{1/z}$ (`@ning2004cycle`, `spec:battery_degradation_v1`). Numerical stability: bounded exponent $1/z \in [0.5, 10]$. Domain of validity: $C_{\text{target}} \in [0, 1], D \in (0, 1]$.
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
