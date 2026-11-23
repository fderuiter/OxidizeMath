//! Built-in configurations and starter presets for [`super::ShannonChannelTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ShannonChannelKind, ShannonToolConfig};

pub fn default_config() -> ShannonToolConfig {
    ShannonToolConfig {
        noise_param: 0.11,
        snr_linear: 15.0,
        bandwidth_hz: 1_000.0,
        renyi_alpha: 2.0,
        channel_kind: ShannonChannelKind::Bsc,
        weights_p: vec![0.5, 0.25, 0.125, 0.125],
        weights_q: vec![0.25, 0.25, 0.25, 0.25],
    }
}

pub fn z_channel_config() -> ShannonToolConfig {
    ShannonToolConfig {
        noise_param: 0.25,
        snr_linear: 10.0,
        bandwidth_hz: 1_000.0,
        renyi_alpha: 0.5,
        channel_kind: ShannonChannelKind::ZChannel,
        weights_p: vec![0.6, 0.3, 0.1],
        weights_q: vec![0.2, 0.5, 0.3],
    }
}

pub fn awgn_hartley_config() -> ShannonToolConfig {
    ShannonToolConfig {
        noise_param: 0.05,
        snr_linear: 31.0,
        bandwidth_hz: 20_000.0,
        renyi_alpha: 2.0,
        channel_kind: ShannonChannelKind::Awgn,
        weights_p: vec![0.7, 0.1, 0.1, 0.1],
        weights_q: vec![0.25, 0.25, 0.25, 0.25],
    }
}

pub const PRESET_Z_CHANNEL_ASYMMETRIC: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Asymmetric Z-Channel Blahut-Arimoto",
    "domain": "algorithmic_information",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "noise_param": 0.25,
    "snr_linear": 10.0,
    "bandwidth_hz": 1000.0,
    "renyi_alpha": 0.5,
    "channel_kind": "ZChannel",
    "weights_p": [0.6, 0.3, 0.1],
    "weights_q": [0.2, 0.5, 0.3]
  }
}"#;

pub const PRESET_AWGN_HARTLEY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Shannon-Hartley AWGN High-SNR Link",
    "domain": "algorithmic_information",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "noise_param": 0.05,
    "snr_linear": 31.0,
    "bandwidth_hz": 20000.0,
    "renyi_alpha": 2.0,
    "channel_kind": "Awgn",
    "weights_p": [0.7, 0.1, 0.1, 0.1],
    "weights_q": [0.25, 0.25, 0.25, 0.25]
  }
}"#;
