//! Domain-specific zero-allocation [`super::frame::SonificationFrame`] extractors across flagship tools.

#[cfg(feature = "pure_math")]
pub mod acoustics;
pub mod medical;
#[cfg(feature = "physics")]
pub mod medical_radar;
#[cfg(feature = "physics")]
pub mod mri;
pub mod neuroscience;
pub mod number_theory;
pub mod quantum_chaos;
pub mod reinforcement_learning;
pub mod spin_resonance;
#[cfg(all(feature = "pure_math", feature = "biology"))]
pub mod surface_morphogenesis;
#[cfg(feature = "pure_math")]
pub mod wave_plate;

#[cfg(feature = "pure_math")]
pub use acoustics::chladni_plate_frame;
pub use medical::{imrt_delivery_frame, radar_linac_frame, LinacGatePhase};
#[cfg(feature = "physics")]
pub use medical_radar::{
    doppler_shift_60ghz_hz, medical_radar_frame, RadarDopplerAudioInput, RADAR_CARRIER_60GHZ_HZ,
};
#[cfg(feature = "physics")]
pub use mri::{
    is_rf_pulse_window, mri_bloch_frame, mri_kspace_frame, mri_kspace_probe_frame,
    mri_pulse_playhead_frame, MriKSpaceAudioInput, MriKSpaceProbeAudioInput,
};
pub use neuroscience::{hodgkin_huxley_frame, neural_ring_frame, spike_analysis_frame};
pub use number_theory::{
    elliptic_torsion_frame, fractran_register_frame, just_intonation_ratio, prime_chord_frame,
    ulam_spiral_frame, EllipticTorsionSample, UlamPolySample,
};
pub use quantum_chaos::{attractor_frame, bifurcation_frame, quantum_wavepacket_frame};
pub use reinforcement_learning::rl_td_surprise_frame;
pub use spin_resonance::{
    bloch_fid_frame, hahn_echo_isochromat_frame, kspace_voxel_ensemble_frame, zeeman_bloch_frame,
};
#[cfg(all(feature = "pure_math", feature = "biology"))]
pub use surface_morphogenesis::surface_morphogenesis_frame;
#[cfg(feature = "pure_math")]
pub use wave_plate::chladni_wave_tank_frame;
