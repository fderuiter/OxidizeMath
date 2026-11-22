#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Shared parameter, state, and snapshot types for [`super::AttractorRunner`].

use super::twin_state::{AttractorSystemKind, ExtraChaosParams, TwinFrame, DEFAULT_EPSILON};
use crate::framework::TimelineRingBuffer;
use nalgebra::Vector3;
use std::collections::VecDeque;

/// Maximum allowed simulation sub-steps per frame for [`super::AttractorPlotter`].
pub const MAX_SIMULATION_SPEED: usize = 50;

/// Execution lifecycle state of [`super::AttractorPlotter`] and [`super::AttractorRunner`].
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SolverExecutionState {
    /// Initialized and ready to run.
    Ready,
    /// Actively integrating frames.
    Running,
    /// Paused by user interaction, single-stepping, or timeline scrubbing.
    Paused,
    /// Halted due to non-finite coordinates or parameters.
    Diverged {
        /// Coordinates observed when divergence occurred.
        #[serde(with = "serde_vec3")]
        invalid_vec: Vector3<f64>,
        /// Step index at which non-finite state was detected.
        diverged_at_step: usize,
    },
}

mod serde_vec3 {
    use super::*;
    use serde::de::{self, SeqAccess, Visitor};
    use serde::ser::SerializeSeq;
    use serde::{Deserializer, Serializer};
    use std::fmt;

    struct F64Def(f64);

    impl serde::Serialize for F64Def {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if self.0.is_nan() {
                serializer.serialize_str("NaN")
            } else if self.0 == f64::INFINITY {
                serializer.serialize_str("Infinity")
            } else if self.0 == f64::NEG_INFINITY {
                serializer.serialize_str("-Infinity")
            } else {
                serializer.serialize_f64(self.0)
            }
        }
    }

    struct F64Visitor;

    impl<'de> Visitor<'de> for F64Visitor {
        type Value = f64;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a float, null, or special float string")
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> {
            Ok(v)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<f64, E> {
            Ok(v as f64)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<f64, E> {
            Ok(v as f64)
        }
        fn visit_unit<E: de::Error>(self) -> Result<f64, E> {
            Ok(f64::NAN)
        }
        fn visit_none<E: de::Error>(self) -> Result<f64, E> {
            Ok(f64::NAN)
        }
        fn visit_str<E: de::Error>(self, s: &str) -> Result<f64, E> {
            match s {
                "NaN" | "nan" => Ok(f64::NAN),
                "Infinity" | "+Infinity" | "inf" | "+inf" => Ok(f64::INFINITY),
                "-Infinity" | "-inf" => Ok(f64::NEG_INFINITY),
                other => other.parse::<f64>().map_err(de::Error::custom),
            }
        }
    }

    pub fn serialize<S: Serializer>(vec: &Vector3<f64>, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(3))?;
        seq.serialize_element(&F64Def(vec.x))?;
        seq.serialize_element(&F64Def(vec.y))?;
        seq.serialize_element(&F64Def(vec.z))?;
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vector3<f64>, D::Error> {
        struct Vec3Visitor;
        impl<'de> Visitor<'de> for Vec3Visitor {
            type Value = Vector3<f64>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a 3-element array of coordinates")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                struct Seed;
                impl<'de> de::DeserializeSeed<'de> for Seed {
                    type Value = f64;
                    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<f64, D::Error> {
                        d.deserialize_any(F64Visitor)
                    }
                }
                let x = seq
                    .next_element_seed(Seed)?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let y = seq
                    .next_element_seed(Seed)?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let z = seq
                    .next_element_seed(Seed)?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                Ok(Vector3::new(x, y, z))
            }
        }
        deserializer.deserialize_seq(Vec3Visitor)
    }
}

/// Shared parameter block read by [`super::AttractorRunner`] without dynamic heap allocation.
#[derive(Debug, Clone, Copy)]
pub struct AttractorSharedParams {
    /// Lorenz Prandtl number $\sigma$.
    pub sigma: f64,
    /// Lorenz Rayleigh number $\rho$.
    pub rho: f64,
    /// Lorenz geometric factor $\beta$.
    pub beta: f64,
    /// Integration step size `dt`.
    pub dt: f64,
    /// Sub-steps executed per worker frame.
    pub simulation_speed: usize,
    /// Maximum rendered trail length (`<= 4,096`).
    pub max_points: usize,
    /// Active dynamical system kind.
    pub system_kind: AttractorSystemKind,
    /// Primary initial coordinates `[c0, c1, c2]`.
    pub initial_state: [f64; 3],
    /// Whether the perturbed shadow trajectory is enabled.
    pub shadow_enabled: bool,
    /// Pre-evaluated AST perturbation offset `[dx0, dy0, dz0]`.
    pub perturbation_offset: [f64; 3],
    /// Auxiliary Rössler and double-pendulum constants.
    pub extra: ExtraChaosParams,
}

impl Default for AttractorSharedParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            dt: 0.01,
            simulation_speed: 5,
            max_points: 2000,
            system_kind: AttractorSystemKind::Lorenz,
            initial_state: [1.0, 1.0, 1.0],
            shadow_enabled: true,
            perturbation_offset: [DEFAULT_EPSILON, 0.0, 0.0],
            extra: ExtraChaosParams::default(),
        }
    }
}

/// Worker snapshot published to the UI thread.
#[derive(Clone)]
pub struct AttractorSnapshot {
    /// Primary 3D trajectory points up to the active timeline cursor.
    pub history: VecDeque<Vector3<f64>>,
    /// Shadow 3D trajectory points up to the active timeline cursor.
    pub shadow_history: VecDeque<Vector3<f64>>,
    /// Full ring buffer of recorded `TwinFrame`s for UI scrubbing.
    pub timeline: TimelineRingBuffer<TwinFrame>,
    /// Current primary 3D phase-space vector.
    pub current_vec: Vector3<f64>,
    /// Current step index at the active timeline cursor.
    pub step_count: usize,
    /// Worker solver state.
    pub execution_state: SolverExecutionState,
}
