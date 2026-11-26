use super::types::*;
use math_commons::constants::liquid_drop_constants;

/// Trait defining nuclear mass-defect binding energy $B(Z, A)$:
/// $$B(Z, A) = \left(Z m_p + (A - Z) m_n - M(Z, A)\right) c^2$$
///
/// Reference: Weizsäcker (1935), DOI: `10.1007/BF01337700`; Bethe & Bacher (1936), DOI:
/// `10.1103/RevModPhys.8.82`; see [`docs/math_specs/domain_physics.md`](../../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// Evaluates $B(Z, A) / A$ directly without singular division because `MassNumber` enforces $A \ge 1$.
///
/// # Domain of Validity
/// Bound or quasi-bound nuclides satisfying $0 \le Z \le A$ and $A \ge 1$.
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

/// Bethe-Weizsäcker Semi-Empirical Mass Formula (Liquid Drop Model):
/// $$B(Z, A) = a_V A - a_S A^{2/3} - a_C \frac{Z(Z - 1)}{A^{1/3}} - a_{\text{sym}} \frac{(A - 2Z)^2}{A} \pm \delta(A, Z)$$
///
/// Reference: Weizsäcker (1935), DOI: `10.1007/BF01337700`; Bethe & Bacher (1936), DOI:
/// `10.1103/RevModPhys.8.82`; see [`docs/math_specs/domain_physics.md`](../../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// Closed-form algebraic evaluation over positive fractional powers $A^{2/3}, A^{-1/3}, A^{-1/2}$;
/// free of catastrophic cancellation away from drip-line extremes.
///
/// # Domain of Validity
/// Medium-to-heavy nuclei ($A \gtrsim 12$, $0 \le Z \le A$) where macroscopic volume/surface scaling
/// dominates magic-number shell corrections.
#[derive(Debug, Clone)]
pub struct LiquidDropModel {
    /// Volume binding coefficient $a_V$ (MeV).
    pub a_v: f64,
    /// Surface tension deficit coefficient $a_S$ (MeV).
    pub a_s: f64,
    /// Coulomb proton-repulsion coefficient $a_C$ (MeV).
    pub a_c: f64,
    /// Pauli asymmetry coefficient $a_{\text{sym}}$ (MeV).
    pub a_sym: f64,
    /// Even-odd nucleon pairing amplitude $a_P$ (MeV) scaling $A^{-1/2}$.
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
}

impl BindingEnergyModel for LiquidDropModel {
    #[verified_engine::verified]
    fn binding_energy(
        &self,
        atomic_number: AtomicNumber,
        mass_number: MassNumber,
    ) -> Result<f64, NuclearError> {
        let z = atomic_number.value() as f64;
        let a = mass_number.value() as f64;

        if atomic_number.value() > mass_number.value() {
            return Err(NuclearError::InvalidAtomicNumber(
                "Z cannot be greater than A".to_string(),
            ));
        }

        let vol_term = self.a_v * a;
        let surf_term = self.a_s * a.powf(2.0 / 3.0);
        let coul_term = self.a_c * (z * (z - 1.0)) / a.powf(1.0 / 3.0);
        let sym_term = self.a_sym * (a - 2.0 * z).powi(2) / a;

        // Pairing term delta
        let z_val = atomic_number.value();
        let n_val = mass_number.value() - z_val;

        let delta = if z_val.is_multiple_of(2) && n_val.is_multiple_of(2) {
            // Even Z, Even N
            self.delta_coeff * a.powf(-0.5)
        } else if !z_val.is_multiple_of(2) && !n_val.is_multiple_of(2) {
            // Odd Z, Odd N
            -self.delta_coeff * a.powf(-0.5)
        } else {
            // Odd A (one even, one odd)
            0.0
        };

        let b = vol_term - surf_term - coul_term - sym_term + delta;
        Ok(b)
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
