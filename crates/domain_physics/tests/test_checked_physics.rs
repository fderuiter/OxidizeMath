//! Integration tests for checked constructors and typed error enums across `domain_physics`.

use domain_physics::error::{
    DoseFluenceError, FluidError, RadarError, SolidStateError, StandardModelError, StatMechError,
};
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::state::LatticeState;
use domain_physics::physics::medical::calibration::hu_to_density;
use domain_physics::physics::medical::optical_motion::calibration::LinearCalibrator;
use domain_physics::physics::medical::radar_gating::mimo::Beamformer;
use domain_physics::physics::medical::radar_gating::tracking::{
    ConstantVelocityModel, TrackingFilter,
};
use domain_physics::physics::mri::proton::boltzmann_ratio;
use domain_physics::physics::solid_state::bcs::coherence_factors;
use domain_physics::physics::solid_state::second_quantization::{Boson, Fermion, FockState};
use domain_physics::physics::standard_model::higgs::HiggsPotential;

#[test]
fn fluid_and_radar_checked_apis_reject_invalid_configurations() {
    assert!(matches!(
        LatticeState::<9>::try_new(usize::MAX, 2),
        Err(FluidError::GridDimensionsOverflow {
            width: usize::MAX,
            height: 2,
        })
    ));
    assert!(LatticeState::<9>::try_new(4, 4).is_ok());

    let model = ConstantVelocityModel::new(0.01, 0.1);
    assert!(matches!(
        TrackingFilter::try_new(1.0, 0.0, model.clone()),
        Err(RadarError::InvalidTrackingParameter {
            parameter: "dt",
            ..
        })
    ));
    assert!(TrackingFilter::try_new(1.0, 0.05, model).is_ok());

    let mut beamformer = Beamformer::new_ula(4, 0.01, 0.02);
    assert_eq!(
        beamformer.set_weights(&[1.0, 1.0]),
        Err(RadarError::AntennaWeightCountMismatch {
            expected: 4,
            actual: 2,
        })
    );
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
        coherence_factors(f64::NAN, 0.2),
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
