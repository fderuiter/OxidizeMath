//! Domain-specific zero-allocation [`super::frame::SonificationFrame`] extractors across flagship tools.

// theory_verification!

#[cfg(feature = "physics")]
pub mod medical_radar;
#[cfg(feature = "physics")]
pub mod mri;
pub mod neuroscience;
pub mod number_theory;
pub mod quantum_chaos;

#[cfg(feature = "physics")]
pub use medical_radar::{
    doppler_shift_60ghz_hz, medical_radar_frame, RadarDopplerAudioInput, RADAR_CARRIER_60GHZ_HZ,
};
#[cfg(feature = "physics")]
pub use mri::{mri_bloch_frame, mri_kspace_frame, mri_pulse_playhead_frame, MriKSpaceAudioInput};
pub use neuroscience::{hodgkin_huxley_frame, neural_ring_frame, spike_analysis_frame};
pub use number_theory::{just_intonation_ratio, prime_chord_frame};
pub use quantum_chaos::{attractor_frame, bifurcation_frame, quantum_wavepacket_frame};
