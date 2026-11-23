//! # Astrophysics & Dwarf Galaxy Dynamics
//!
//! Empirical dwarf-galaxy regressions from Tillaboev, Tadjibaev & Otojanova (2025)
//! (`tillaboev2025`), radially decomposed baryonic and dark-matter (`NfwCusp`,
//! `BurkertCore`) circular rotation curves, and zero-allocation kick-drift-kick
//! symplectic planar disk and tidal-stream orbits.

pub mod galaxies;
pub mod orbits;
pub mod rotation_curves;

pub use galaxies::{
    GalaxyModel, GeneralIrregular, Magnitude, Mpc, Redshift, SolarMassLog, TypeCode10,
    TypeCode95To99,
};
pub use orbits::{
    GalacticDiskSystem, KPC_PER_MYR_PER_KMS, OrbitParticleKind, OrbitPopulationConfig,
    OrbitalInvariants, OrbitalState, SECONDS_PER_MYR, planar_acceleration,
};
pub use rotation_curves::{
    BaryonicDiskConfig, DwarfMorphology, G_KPC_KMS2_PER_SOLAR_MASS, HaloProfile, METERS_PER_KPC,
    MIN_GALACTOCENTRIC_RADIUS_KPC, TillaboevScalingSummary, VelocityDecomposition,
    baryons_from_absolute_magnitude, baryons_from_distance, clamp_radius_kpc,
    decompose_rotation_curve, sample_rotation_curve_into, total_gravitational_potential,
};

// [cite:dwarf_galaxy_empirical_dependencies]

use pure_math::theory_verification;

theory_verification!(
    module = "astrophysics",
    constants = {
        V_DISK_KMS = 30.0;
        V_GAS_KMS = 40.0;
        V_HALO_KMS = 120.0;
        EXPECTED_V_CIRC_KMS = 130.0;
    },
    test = {
        use approx::assert_relative_eq;
        let v_circ =
            (V_DISK_KMS * V_DISK_KMS + V_GAS_KMS * V_GAS_KMS + V_HALO_KMS * V_HALO_KMS).sqrt();
        assert_relative_eq!(
            v_circ,
            EXPECTED_V_CIRC_KMS,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
