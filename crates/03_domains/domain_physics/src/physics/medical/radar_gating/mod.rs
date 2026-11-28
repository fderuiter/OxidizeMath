//! # Clinical Respiratory Beam Gating
//!
//! Clinical decision-layer state machine (`GatingLogic`, `BeamState`, `GatingError`) for linear
//! accelerator (LINAC) respiratory beam gating with Schmidt-trigger hysteresis, hardware latency
//! compensation, and IEEE 754 `NaN`/`Inf` safety interlock latching.
//!
//! Raw RF/mmWave FMCW radar signal processing, Bluestein CZT spectral zoom, MIMO beamforming,
//! MUSIC super-resolution, and Kalman tracking reside in `domain_engineering::engineering::radar`.

pub mod gating;

pub use gating::{BeamState, GatingError, GatingLogic};

// [cite:mmwave_radiotherapy_setup]
