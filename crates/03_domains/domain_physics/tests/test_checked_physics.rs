//! Integration tests for checked constructors and typed error enums across `domain_physics`.

use domain_physics::error::{
    DoseFluenceError, FluidError, QuantumError, SolidStateError, StandardModelError, StatMechError,
};
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::state::LatticeState;
use domain_physics::physics::medical::calibration::hu_to_density;
use domain_physics::physics::medical::optical_motion::calibration::LinearCalibrator;
use domain_physics::physics::mri::proton::boltzmann_ratio;
use domain_physics::physics::quantum::OrbitalQuantumNumbers;
use domain_physics::physics::solid_state::bcs::coherence_factors_strong;
use domain_physics::physics::solid_state::second_quantization::{Boson, Fermion, FockState};
use domain_physics::physics::solid_state::types::ElectronVolts;
use domain_physics::physics::standard_model::higgs::HiggsPotential;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn fluid_checked_apis_reject_invalid_configurations() {
    assert!(matches!(
        LatticeState::<9>::try_new(usize::MAX, 2),
        Err(FluidError::GridDimensionsOverflow {
            width: usize::MAX,
            height: 2,
        })
    ));
    assert!(LatticeState::<9>::try_new(4, 4).is_ok());
}

#[test]
fn medical_mri_solid_state_and_higgs_return_typed_domain_errors() {
    assert!(matches!(
        hu_to_density(-1500.0),
        Err(DoseFluenceError::Math(_))
    ));
    assert_eq!(
        LinearCalibrator::fit(&[1.0], &[2.0]).unwrap_err(),
        DoseFluenceError::InsufficientCalibrationPoints {
            minimum: 2,
            actual: 1,
        }
    );
    assert_eq!(
        LinearCalibrator::fit(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]).unwrap_err(),
        DoseFluenceError::ZeroCalibrationVariance
    );

    assert_eq!(
        boltzmann_ratio(0.0, 3.0),
        Err(StatMechError::InvalidTemperature(0.0))
    );
    assert!(matches!(
        coherence_factors_strong(ElectronVolts::new(f64::NAN), ElectronVolts::new(0.2)),
        Err(SolidStateError::Math(_))
    ));

    let mut fock = FockState::new(2);
    assert_eq!(
        fock.set_occupation::<Fermion>(5, 1),
        Err(SolidStateError::FockIndexOutOfBounds { index: 5, len: 2 })
    );
    assert_eq!(
        fock.set_occupation::<Fermion>(0, 2),
        Err(SolidStateError::PauliExclusionViolation { requested: 2 })
    );
    fock.set_occupation::<Boson>(1, u8::MAX).unwrap();
    assert_eq!(
        fock.create_particle::<Boson>(1),
        Err(SolidStateError::BosonOccupationOverflow { max: u8::MAX })
    );

    let unbroken = HiggsPotential {
        mu2: 1.0,
        lambda: 0.5,
    };
    assert_eq!(
        unbroken.vev(),
        Err(StandardModelError::UnbrokenSymmetry { mu2: 1.0 })
    );
    let unstable = HiggsPotential {
        mu2: -1.0,
        lambda: 0.0,
    };
    assert_eq!(
        unstable.vev(),
        Err(StandardModelError::UnstableHiggsPotential { lambda: 0.0 })
    );
}

#[test]
fn orbital_quantum_numbers_new_returns_typed_quantum_errors_without_allocating() {
    let (ok, _) = VerifiedEngine::run_verified(|| {
        assert_eq!(
            OrbitalQuantumNumbers::new(0, 0, 0),
            Err(QuantumError::InvalidPrincipalQuantumNumber {
                n: 0,
                min: 1,
                max: 6,
            })
        );
        assert_eq!(
            OrbitalQuantumNumbers::new(7, 0, 0),
            Err(QuantumError::InvalidPrincipalQuantumNumber {
                n: 7,
                min: 1,
                max: 6,
            })
        );
        assert_eq!(
            OrbitalQuantumNumbers::new(2, 2, 0),
            Err(QuantumError::InvalidOrbitalQuantumNumber { l: 2, n: 2 })
        );
        assert_eq!(
            OrbitalQuantumNumbers::new(3, 1, -2),
            Err(QuantumError::InvalidMagneticQuantumNumber { m: -2, l: 1 })
        );
        OrbitalQuantumNumbers::new(3, 2, -1).is_ok()
    });
    assert!(ok);
}
