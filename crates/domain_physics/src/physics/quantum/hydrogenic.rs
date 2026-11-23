//! 3D Hydrogenic Eigenstates and Zero-Allocation Two-State Superposition Evolution.
// [cite:quantum_mechanics]

use crate::physics::quantum::hamiltonian::construct_two_state_eigen_hamiltonian;
use crate::physics::quantum::schrodinger::{evolve_state_in_place, time_evolution_operator};
use crate::physics::quantum::types::{QuantumOperator, QuantumState};
use math_commons::math_kernel::types::flatten_3d_index;
use nalgebra::DVector;
use num_complex::Complex;
use pure_math::pure_math::special_functions::legendre::real_spherical_harmonic;
use pure_math::pure_math::special_functions::polynomials::associated_laguerre;

/// Validated hydrogenic orbital quantum numbers $(n, \ell, m)$ with $1 \le n \le 6$, $\ell < n$, $|m| \le \ell$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrbitalQuantumNumbers {
    /// Principal quantum number $n \ge 1$.
    pub n: u32,
    /// Azimuthal angular momentum quantum number $0 \le \ell < n$.
    pub l: u32,
    /// Magnetic quantum number $-\ell \le m \le \ell$.
    pub m: i32,
}

impl OrbitalQuantumNumbers {
    /// Validates and constructs `(n, l, m)` within interactive bounds `1 <= n <= 6`.
    #[verified_engine::verified]
    pub fn new(n: u32, l: u32, m: i32) -> Result<Self, String> {
        if !(1..=6).contains(&n) {
            return Err(format!("Principal quantum number n={n} must be in 1..=6"));
        }
        if l >= n {
            return Err(format!(
                "Orbital quantum number l={l} must satisfy l < n={n}"
            ));
        }
        if m.unsigned_abs() > l {
            return Err(format!(
                "Magnetic quantum number m={m} must satisfy |m| <= l={l}"
            ));
        }
        Ok(Self { n, l, m })
    }

    /// Clamps arbitrary `(n, l, m)` into a valid hydrogenic state within `1..=6`.
    #[verified_engine::verified]
    pub fn clamped(n: u32, l: u32, m: i32) -> Self {
        let safe_n = n.clamp(1, 6);
        let safe_l = l.min(safe_n - 1);
        let limit_m = safe_l as i32;
        let safe_m = m.clamp(-limit_m, limit_m);
        Self {
            n: safe_n,
            l: safe_l,
            m: safe_m,
        }
    }
}

/// Scalar field observable evaluated from a complex 3D superposition $\Psi(\mathbf{r}, t)$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantumFieldObservable {
    /// Signed real wavefunction $\text{Re}\,\Psi(\mathbf{r}, t)$ exposing positive/negative nodal shells.
    SignedRealWavefunction,
    /// Born probability density $|\Psi(\mathbf{r}, t)|^2 \ge 0$.
    ProbabilityDensity,
}

/// Non-relativistic hydrogenic eigenenergy $E_n = -\frac{1}{2n^2}$ in Hartree atomic units ($\hbar = m_e = e = 1$).
#[verified_engine::verified]
pub fn hydrogenic_energy_au(n: u32) -> f64 {
    let nf = (n.max(1)) as f64;
    -0.5 / (nf * nf)
}

#[verified_engine::verified]
fn factorial_u64(k: u64) -> f64 {
    let mut acc = 1.0;
    for i in 2..=k {
        acc *= i as f64;
    }
    acc
}

/// Normalized hydrogenic radial wavefunction $R_{n\ell}(r)$ evaluated using [`associated_laguerre`].
///
/// Satisfies $\int_0^\infty |R_{n\ell}(r)|^2 r^2 \, dr = 1$.
#[verified_engine::verified]
pub fn hydrogenic_radial_wavefunction(n: u32, l: u32, r: f64, bohr_radius: f64) -> f64 {
    if n == 0
        || l >= n
        || !r.is_finite()
        || r < 0.0
        || !bohr_radius.is_finite()
        || bohr_radius <= 0.0
    {
        return 0.0;
    }
    let nf = n as f64;
    let scale = 2.0 / (nf * bohr_radius);
    let rho = scale * r;
    let radial_degree = (n - l - 1) as u64;
    let alpha = (2 * l + 1) as f64;
    let num = factorial_u64(radial_degree);
    let den = 2.0 * nf * factorial_u64((n + l) as u64);
    let prefactor = (scale * scale * scale * (num / den)).sqrt();
    let lag = associated_laguerre(radial_degree, alpha, rho);
    prefactor * (-0.5 * rho).exp() * rho.powi(l as i32) * lag
}

/// Real orthonormal 3D hydrogenic eigenstate $\psi_{n\ell m}(x, y, z) = R_{n\ell}(r)\,Y_{\ell m}(\theta, \phi)$.
#[verified_engine::verified]
pub fn hydrogenic_orbital_cartesian(
    orbital: OrbitalQuantumNumbers,
    x: f64,
    y: f64,
    z: f64,
    bohr_radius: f64,
) -> f64 {
    let r_sq = x * x + y * y + z * z;
    let r = r_sq.sqrt();
    let radial = hydrogenic_radial_wavefunction(orbital.n, orbital.l, r, bohr_radius);
    let (cos_theta, phi) = if r > 1e-12 {
        ((z / r).clamp(-1.0, 1.0), y.atan2(x))
    } else {
        (1.0, 0.0)
    };
    let angular = real_spherical_harmonic(orbital.l as u64, orbital.m as i64, cos_theta, phi);
    radial * angular
}

/// Zero-allocation 3D two-state hydrogenic superposition engine coupled to `schrodinger` and `hamiltonian`.
pub struct HydrogenicSuperposition3D {
    orbital_a: OrbitalQuantumNumbers,
    orbital_b: OrbitalQuantumNumbers,
    grid_dim: usize,
    half_extent: f64,
    bohr_radius: f64,
    dt: f64,
    dipole_coupling: f64,
    /// Two-component Hilbert state $[c_1(t), c_2(t)]^T$.
    pub state: QuantumState,
    hamiltonian: QuantumOperator,
    propagator: QuantumOperator,
    scratch: DVector<Complex<f64>>,
    basis_a: Vec<f64>,
    basis_b: Vec<f64>,
    elapsed_time: f64,
}

impl HydrogenicSuperposition3D {
    /// Pre-allocates basis buffers up to `max_grid_dim^3` and initializes the two-state propagator.
    #[verified_engine::verified]
    pub fn new(
        orbital_a: OrbitalQuantumNumbers,
        orbital_b: OrbitalQuantumNumbers,
        grid_dim: usize,
        max_grid_dim: usize,
        half_extent: f64,
        dt: f64,
    ) -> Self {
        let cap = max_grid_dim.max(grid_dim).pow(3);
        let hamiltonian = construct_two_state_eigen_hamiltonian(
            hydrogenic_energy_au(orbital_a.n),
            hydrogenic_energy_au(orbital_b.n),
            0.0,
        );
        let propagator = time_evolution_operator(&hamiltonian, dt, 1.0);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let state = QuantumState::new(DVector::from_vec(vec![
            Complex::new(inv_sqrt2, 0.0),
            Complex::new(inv_sqrt2, 0.0),
        ]));
        let mut engine = Self {
            orbital_a,
            orbital_b,
            grid_dim: 0,
            half_extent,
            bohr_radius: 1.0,
            dt,
            dipole_coupling: 0.0,
            state,
            hamiltonian,
            propagator,
            scratch: DVector::from_element(2, Complex::new(0.0, 0.0)),
            basis_a: Vec::with_capacity(cap),
            basis_b: Vec::with_capacity(cap),
            elapsed_time: 0.0,
        };
        engine.reconfigure_basis(orbital_a, orbital_b, grid_dim, half_extent, 1.0);
        engine
    }

    /// Recomputes spatial eigenstate grids in-place when quantum numbers or grid geometry change.
    #[verified_engine::verified]
    pub fn reconfigure_basis(
        &mut self,
        orbital_a: OrbitalQuantumNumbers,
        orbital_b: OrbitalQuantumNumbers,
        grid_dim: usize,
        half_extent: f64,
        bohr_radius: f64,
    ) {
        let dim = grid_dim.max(2);
        let unchanged = self.orbital_a == orbital_a
            && self.orbital_b == orbital_b
            && self.grid_dim == dim
            && (self.half_extent - half_extent).abs() < 1e-12
            && (self.bohr_radius - bohr_radius).abs() < 1e-12;
        if unchanged {
            return;
        }
        self.orbital_a = orbital_a;
        self.orbital_b = orbital_b;
        self.grid_dim = dim;
        self.half_extent = half_extent.max(0.5);
        self.bohr_radius = bohr_radius.max(0.1);
        let total = dim * dim * dim;
        self.basis_a.resize(total, 0.0);
        self.basis_b.resize(total, 0.0);
        let step = (2.0 * self.half_extent) / ((dim - 1) as f64);
        for z in 0..dim {
            let wz = -self.half_extent + (z as f64) * step;
            for y in 0..dim {
                let wy = -self.half_extent + (y as f64) * step;
                for x in 0..dim {
                    let wx = -self.half_extent + (x as f64) * step;
                    let idx = flatten_3d_index(x, y, z, dim, dim);
                    self.basis_a[idx] =
                        hydrogenic_orbital_cartesian(orbital_a, wx, wy, wz, self.bohr_radius);
                    self.basis_b[idx] =
                        hydrogenic_orbital_cartesian(orbital_b, wx, wy, wz, self.bohr_radius);
                }
            }
        }
        self.rebuild_propagator(self.dt, self.dipole_coupling);
    }

    /// Updates `dt` and optional dipole coupling, refreshing the $2 \times 2$ unitary propagator.
    #[verified_engine::verified]
    pub fn rebuild_propagator(&mut self, dt: f64, dipole_coupling: f64) {
        self.dt = dt;
        self.dipole_coupling = dipole_coupling;
        self.hamiltonian = construct_two_state_eigen_hamiltonian(
            hydrogenic_energy_au(self.orbital_a.n),
            hydrogenic_energy_au(self.orbital_b.n),
            dipole_coupling,
        );
        self.propagator = time_evolution_operator(&self.hamiltonian, dt, 1.0);
    }

    /// Sets normalized superposition weights `mix` $\in [0, 1]$ (`0` = pure state 1, `1` = pure state 2) and phase.
    #[verified_engine::verified]
    pub fn set_superposition(&mut self, mix: f64, relative_phase: f64) {
        let m = mix.clamp(0.0, 1.0);
        let w_a = (1.0 - m).sqrt();
        let w_b = m.sqrt();
        let (sin_p, cos_p) = relative_phase.sin_cos();
        self.state.vector[0] = Complex::new(w_a, 0.0);
        self.state.vector[1] = Complex::new(w_b * cos_p, w_b * sin_p);
        self.state.normalize_in_place();
        self.elapsed_time = 0.0;
    }

    /// Returns elapsed simulation time in atomic units.
    #[verified_engine::verified]
    pub fn elapsed_time(&self) -> f64 {
        self.elapsed_time
    }

    /// Advances the two-state complex amplitudes by `dt` with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_in_place(&mut self) {
        evolve_state_in_place(&mut self.state, &self.propagator, &mut self.scratch);
        self.state.normalize_in_place();
        self.elapsed_time += self.dt;
    }

    /// Evaluates the 3D superposition observable into `out_voxels` with zero heap allocations.
    #[verified_engine::verified]
    pub fn write_field_into(&self, observable: QuantumFieldObservable, out_voxels: &mut [f32]) {
        let c_a = self.state.vector[0];
        let c_b = self.state.vector[1];
        let zipped = out_voxels
            .iter_mut()
            .zip(self.basis_a.iter().zip(self.basis_b.iter()));
        match observable {
            QuantumFieldObservable::SignedRealWavefunction => {
                for (out, (&ba, &bb)) in zipped {
                    let re = c_a.re * ba + c_b.re * bb;
                    *out = re as f32;
                }
            }
            QuantumFieldObservable::ProbabilityDensity => {
                for (out, (&ba, &bb)) in zipped {
                    let psi = c_a * ba + c_b * bb;
                    *out = psi.norm_sqr() as f32;
                }
            }
        }
    }
}
