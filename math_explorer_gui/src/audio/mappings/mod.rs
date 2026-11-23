//! Domain-specific zero-allocation [`super::frame::SonificationFrame`] extractors across seven flagship tools.

// theory_verification!

pub mod neuroscience;
pub mod number_theory;
pub mod quantum_chaos;

pub use neuroscience::{hodgkin_huxley_frame, neural_ring_frame, spike_analysis_frame};
pub use number_theory::{just_intonation_ratio, prime_chord_frame};
pub use quantum_chaos::{attractor_frame, bifurcation_frame, quantum_wavepacket_frame};
