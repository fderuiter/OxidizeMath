//! Literature-calibrated lithium-ion cell chemistry presets.
// [cite:battery_degradation]

/// Canonical lithium-ion cell chemistry presets with literature-calibrated kinetic parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChemistryPreset {
    /// Nickel-rich $\text{LiNi}_{0.8}\text{Mn}_{0.1}\text{Co}_{0.1}\text{O}_2$ / graphite high-energy cell.
    #[default]
    Nmc811,
    /// Olivine $\text{LiFePO}_4$ / graphite long-cycle-life cell.
    Lfp,
    /// $\text{LiNi}_{0.8}\text{Co}_{0.15}\text{Al}_{0.05}\text{O}_2$ / graphite-Si high-power cell.
    Nca,
    /// Spinel $\text{Li}_4\text{Ti}_5\text{O}_{12}$ zero-strain anode ultra-long-life cell.
    Lto,
}

impl ChemistryPreset {
    /// Ordered list of all supported chemistry presets.
    pub const ALL: [Self; 4] = [Self::Nmc811, Self::Lfp, Self::Nca, Self::Lto];

    /// Short display label for UI selectors and exported cell cards.
    #[must_use]
    #[verified_engine::verified]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nmc811 => "NMC811",
            Self::Lfp => "LFP",
            Self::Nca => "NCA",
            Self::Lto => "LTO",
        }
    }

    /// Concise electrochemical description of cathode/anode pairing.
    #[must_use]
    #[verified_engine::verified]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Nmc811 => "Nickel-rich NMC811 / Graphite (high energy, moderate thermal aging)",
            Self::Lfp => "Lithium Iron Phosphate / Graphite (extended life, diffusion SEI z=0.55)",
            Self::Nca => "Nickel-Cobalt-Aluminum / Graphite-Si (high power, thermal sensitive)",
            Self::Lto => "Lithium Titanate Spinel Anode (ultra-low SEI, fast-charge tolerant)",
        }
    }

    /// Reference DoD power-law prefactor $\alpha$ at $25\,^\circ\text{C}$ and $1\text{C}$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn default_alpha(self) -> f64 {
        match self {
            Self::Nmc811 => 2.20e5,
            Self::Lfp => 5.80e5,
            Self::Nca => 1.85e5,
            Self::Lto => 1.60e6,
        }
    }

    /// Reference DoD power-law exponent $\beta < 0$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn default_beta(self) -> f64 {
        match self {
            Self::Nmc811 => -1.26,
            Self::Lfp => -1.15,
            Self::Nca => -1.30,
            Self::Lto => -0.95,
        }
    }

    /// Default stretched cycling exponent $z \in [0.1, 2.0]$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn default_exponent_z(self) -> f64 {
        match self {
            Self::Nmc811 => 0.60,
            Self::Lfp => 0.55,
            Self::Nca => 0.65,
            Self::Lto => 0.75,
        }
    }

    /// Arrhenius activation energy $E_a$ in $\text{J}\cdot\text{mol}^{-1}$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn default_activation_energy(self) -> f64 {
        match self {
            Self::Nmc811 => 31_500.0,
            Self::Lfp => 24_500.0,
            Self::Nca => 34_000.0,
            Self::Lto => 18_000.0,
        }
    }

    /// C-rate stress exponent $\gamma_c$ scaling effective cycle damage as $I_c^{\gamma_c}$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn default_c_rate_exponent(self) -> f64 {
        match self {
            Self::Nmc811 => 0.55,
            Self::Lfp => 0.35,
            Self::Nca => 0.65,
            Self::Lto => 0.18,
        }
    }

    /// Parabolic SEI calendar fade coefficient $k_{\text{SEI}}$ in $\text{day}^{-1/2}$ at $25\,^\circ\text{C}$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn default_sei_rate(self) -> f64 {
        match self {
            Self::Nmc811 => 2.2e-3,
            Self::Lfp => 1.4e-3,
            Self::Nca => 2.6e-3,
            Self::Lto => 4.5e-4,
        }
    }
}
