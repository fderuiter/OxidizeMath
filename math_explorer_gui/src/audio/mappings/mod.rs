//! Domain-specific zero-allocation [`super::frame::SonificationFrame`] extractors across flagship tools.

// theory_verification!

pub mod neuroscience;
pub mod number_theory;
pub mod quantum_chaos;
pub mod spin_resonance;

pub use neuroscience::{hodgkin_huxley_frame, neural_ring_frame, spike_analysis_frame};
pub use number_theory::{just_intonation_ratio, prime_chord_frame};
pub use quantum_chaos::{attractor_frame, bifurcation_frame, quantum_wavepacket_frame};
pub use spin_resonance::{
    bloch_fid_frame, hahn_echo_isochromat_frame, kspace_voxel_ensemble_frame, zeeman_bloch_frame,
};
