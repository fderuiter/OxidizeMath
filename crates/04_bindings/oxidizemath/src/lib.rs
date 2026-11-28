#![doc = include_str!("../README.md")]

pub mod error;
pub mod prelude;

pub use diagnostics;
pub use engine_diffeq;
pub use engine_linalg;
pub use engine_opt;
pub use engine_opt::fit;
pub use engine_pde;
pub use math_commons::{
    ast, constants, math_kernel, poi, primitives, registry, series, stochastic, wigner,
};
pub use math_memory;
pub use math_scalars;
pub use scientific_metadata;
pub use verified_engine;

#[cfg(feature = "ai")]
pub use domain_ai::ai;

#[cfg(feature = "applied")]
pub mod applied {
    //! Applied mathematics, scientific computing, finance, and clinical statistics.
    pub use ::domain_applied::applied::{
        algorithms, cannibalism, clinical_trials, error, favoritism, freesurfer, game_theory, grpo,
        isosurface, lorahub, pharmacokinetics, portfolio, win_ratio,
    };
    #[cfg(feature = "engineering")]
    pub use ::domain_engineering::engineering::battery_degradation;
    /// Closed-loop control plants and (with `engineering`) manufacturing and embedded calculators.
    pub mod engineering {
        pub use ::domain_applied::applied::engineering::*;
        #[cfg(feature = "engineering")]
        pub use ::domain_engineering::engineering::cnc::*;
        #[cfg(feature = "engineering")]
        pub use ::domain_engineering::engineering::uart::*;
        #[cfg(feature = "engineering")]
        pub use ::domain_engineering::engineering::{cnc, uart};
    }
}

#[cfg(feature = "biology")]
pub use domain_biology::biology;

#[cfg(feature = "climate")]
pub use domain_climate::climate;

#[cfg(feature = "engineering")]
pub use domain_engineering::engineering;

#[cfg(feature = "epidemiology")]
pub use domain_epidemiology::epidemiology;

#[cfg(feature = "physics")]
pub mod physics {
    //! Classical, quantum, relativistic, statistical, and medical physics.
    pub use ::domain_physics::physics::{
        astrophysics, chaos, error, fluid_dynamics, high_energy, mri, nuclear, optics, quantum,
        solid_state, standard_model, stat_mech,
    };
    /// Medical physics, dosimetry, optical motion, and clinical respiratory gating.
    pub mod medical {
        pub use ::domain_physics::physics::medical::{
            GatingError, accelerator, calibration, dose, evaluation, imaging, optical_motion,
            optimization, signal, thermodynamics,
        };
        /// Clinical respiratory beam gating and (with `engineering`) RF/mmWave radar solvers.
        pub mod radar_gating {
            #[cfg(feature = "engineering")]
            pub use ::domain_engineering::engineering::radar::*;
            pub use ::domain_physics::physics::medical::radar_gating::*;
        }
    }
}

#[cfg(feature = "pure_math")]
pub use ::pure_math::math_types;
#[cfg(feature = "pure_math")]
pub use ::pure_math::pure_math::statistics::{bootstrap, monte_carlo, qmc};
#[cfg(feature = "pure_math")]
#[path = "pure_math_facade.rs"]
pub mod pure_math;
#[cfg(feature = "pure_math")]
pub use nalgebra;
#[cfg(feature = "pure_math")]
pub use num_complex;
#[cfg(feature = "pure_math")]
pub use oxidize_core::{mesh, rng};
#[cfg(feature = "pure_math")]
pub use petgraph;
#[cfg(feature = "pure_math")]
pub use rand;
