//! Domain-specific zero-allocation [`super::frame::SonificationFrame`] extractors across flagship tools.

// theory_verification!

#[cfg(feature = "pure_math")]
pub mod acoustics;
pub mod neuroscience;
pub mod number_theory;
pub mod quantum_chaos;

#[cfg(feature = "pure_math")]
pub use acoustics::chladni_plate_frame;
pub use neuroscience::{hodgkin_huxley_frame, neural_ring_frame, spike_analysis_frame};
pub use number_theory::{just_intonation_ratio, prime_chord_frame};
pub use quantum_chaos::{attractor_frame, bifurcation_frame, quantum_wavepacket_frame};
