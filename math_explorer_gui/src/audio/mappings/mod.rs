//! Domain-specific zero-allocation [`super::frame::SonificationFrame`] extractors across flagship tools.

// theory_verification!

pub mod neuroscience;
pub mod number_theory;
pub mod quantum_chaos;
#[cfg(feature = "pure_math")]
pub mod wave_plate;

pub use neuroscience::{hodgkin_huxley_frame, neural_ring_frame, spike_analysis_frame};
pub use number_theory::{just_intonation_ratio, prime_chord_frame};
pub use quantum_chaos::{attractor_frame, bifurcation_frame, quantum_wavepacket_frame};
#[cfg(feature = "pure_math")]
pub use wave_plate::chladni_wave_tank_frame;
