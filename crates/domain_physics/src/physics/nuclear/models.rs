use super::types::*;
use math_commons::constants::liquid_drop_constants;

/// Trait defining the behavior of a nuclear binding energy model.
pub trait BindingEnergyModel {
    /// Calculates the Binding Energy B(Z, A).
    #[verified_engine::verified]
    fn binding_energy(
        &self,
        atomic_number: AtomicNumber,
        mass_number: MassNumber,
    ) -> Result<f64, NuclearError>;

    /// Calculates the Binding Energy per nucleon.
    #[verified_engine::verified]
    fn binding_energy_per_nucleon(
        &self,
        atomic_number: AtomicNumber,
        mass_number: MassNumber,
    ) -> Result<f64, NuclearError> {
        let be = self.binding_energy(atomic_number, mass_number)?;
        Ok(be / mass_number.value() as f64)
    }
}

/// Signed term-by-term breakdown of the Weizsäcker Semi-Empirical Mass Formula (in MeV).
///
/// All five physical terms carry their signed contribution to total binding energy $B(Z, A)$:
/// `volume + surface + coulomb + asymmetry + pairing == total`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidDropBreakdown {
    /// Positive bulk nuclear attraction $+a_V A$ (MeV).
    pub volume: f64,
    /// Negative surface tension correction $-a_S A^{2/3}$ (MeV).
    pub surface: f64,
    /// Negative proton-proton electrostatic repulsion $-a_C Z(Z - 1) A^{-1/3}$ (MeV).
    pub coulomb: f64,
    /// Negative Pauli neutron-proton asymmetry penalty $-a_{\text{sym}} (A - 2Z)^2 / A$ (MeV).
    pub asymmetry: f64,
    /// Signed nucleon pairing contribution $+\delta_0 A^{-1/2}$ (even-even), $-\delta_0 A^{-1/2}$ (odd-odd), or $0$ (odd $A$).
    pub pairing: f64,
    /// Total nuclear binding energy $B(Z, A)$ (MeV).
    pub total: f64,
    /// Binding energy per nucleon $B(Z, A) / A$ (MeV).
    pub per_nucleon: f64,
}

/// The Liquid Drop Model (Semi-Empirical Mass Formula).
#[derive(Debug, Clone)]
pub struct LiquidDropModel {
    #[allow(missing_docs)]
    pub a_v: f64,
    #[allow(missing_docs)]
    pub a_s: f64,
    #[allow(missing_docs)]
    pub a_c: f64,
    #[allow(missing_docs)]
    pub a_sym: f64,
    #[allow(missing_docs)]
    pub delta_coeff: f64,
}

impl Default for LiquidDropModel {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            a_v: liquid_drop_constants::A_V,
            a_s: liquid_drop_constants::A_S,
            a_c: liquid_drop_constants::A_C,
            a_sym: liquid_drop_constants::A_SYM,
            delta_coeff: liquid_drop_constants::DELTA_COEFF,
        }
    }
}

impl LiquidDropModel {
    /// Creates a new LiquidDropModel with standard constants.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new LiquidDropModel with custom constants.
    #[verified_engine::verified]
    pub fn with_constants(a_v: f64, a_s: f64, a_c: f64, a_sym: f64, delta_coeff: f64) -> Self {
        Self {
            a_v,
            a_s,
            a_c,
            a_sym,
            delta_coeff,
        }
    }

    /// Computes the full Weizsäcker binding-energy term breakdown for nuclide $(Z, A)$.
    #[verified_engine::verified]
    pub fn binding_energy_breakdown(
        &self,
        atomic_number: AtomicNumber,
        mass_number: MassNumber,
    ) -> Result<LiquidDropBreakdown, NuclearError> {
        let z_val = atomic_number.value();
        let a_val = mass_number.value();
        if z_val > a_val {
            return Err(NuclearError::InvalidAtomicNumber(
                "Z cannot be greater than A".to_string(),
            ));
        }

        let z = z_val as f64;
        let a = a_val as f64;
        let volume = self.a_v * a;
        let surface = -self.a_s * a.powf(2.0 / 3.0);
        let coulomb = -self.a_c * (z * (z - 1.0)) / a.powf(1.0 / 3.0);
        let asymmetry = -self.a_sym * (a - 2.0 * z).powi(2) / a;

        let n_val = a_val - z_val;
        let pairing = if z_val.is_multiple_of(2) && n_val.is_multiple_of(2) {
            self.delta_coeff * a.powf(-0.5)
        } else if !z_val.is_multiple_of(2) && !n_val.is_multiple_of(2) {
            -self.delta_coeff * a.powf(-0.5)
        } else {
            0.0
        };

        let total = volume + surface + coulomb + asymmetry + pairing;
        Ok(LiquidDropBreakdown {
            volume,
            surface,
            coulomb,
            asymmetry,
            pairing,
            total,
            per_nucleon: total / a,
        })
    }
}

impl BindingEnergyModel for LiquidDropModel {
    #[verified_engine::verified]
    fn binding_energy(
        &self,
        atomic_number: AtomicNumber,
        mass_number: MassNumber,
    ) -> Result<f64, NuclearError> {
        self.binding_energy_breakdown(atomic_number, mass_number)
            .map(|b| b.total)
    }
}

/// The Shell Model (Spin-Orbit coupling).
pub mod shell {
    use math_commons::constants;

    /// Calculates the spin-orbit expectation value <L.S>.
    ///
    /// Formula: <L.S> = (hbar^2 / 2) * (j(j+1) - l(l+1) - s(s+1))
    ///
    /// # Arguments
    /// * `l` - Orbital angular momentum quantum number.
    /// * `s` - Spin angular momentum quantum number.
    /// * `j` - Total angular momentum quantum number.
    ///
    /// # Returns
    /// * `f64` - The energy shift factor.
    #[verified_engine::verified]
    pub fn spin_orbit_coupling(l: f64, s: f64, j: f64) -> f64 {
        let hbar = constants::HBAR_C / constants::LIGHT_SPEED;
        let term = j * (j + 1.0) - l * (l + 1.0) - s * (s + 1.0);
        (hbar.powi(2) / 2.0) * term
    }
}
