use verified_engine::engine::VerifiedEngine;

use super::backend::SonificationBackend;
use super::frame::LIMITER_CEILING_LINEAR;
use super::mappings::number_theory::{
    just_intonation_rational, prime_chord_frame, PRIME_CHORD_ROOT_HZ,
};
use super::midi::{
    decode_vlq, encode_frames_to_midi, encode_midi_voice_tracks, encode_vlq, seconds_to_ticks,
    MidiNoteSpan, MidiPitch, MIDI_TICKS_PER_QUARTER,
};
use super::scala::{ScalaPitch, ScalaScale};
use super::synth::OfflinePcmSynthesizer;
use super::wav::{
    render_offline_pcm_into, synthesize_offline_wav, OfflineRenderConfig, WAV_EXPORT_SAMPLE_RATE,
    WAV_STUDIO_SAMPLE_RATE,
};
use super::WaveformPreset;

fn parse_and_validate_scl(scl: &str) -> Vec<f64> {
    let mut non_comments = scl
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('!') && !l.is_empty());
    let desc = non_comments.next().expect("Scala description line");
    assert!(!desc.is_empty());
    let count: usize = non_comments
        .next()
        .expect("Scala note count line")
        .parse()
        .expect("valid note count");
    assert!(count >= 1);

    let mut cents_out = Vec::with_capacity(count);
    for line in non_comments {
        let token = line.split_whitespace().next().expect("pitch token");
        let cents = if token.contains('.') {
            assert!(!token.contains('/'), "cents token must not contain '/'");
            token.parse::<f64>().expect("valid cents float")
        } else if let Some((n_s, d_s)) = token.split_once('/') {
            let n: u64 = n_s.parse().expect("ratio numerator");
            let d: u64 = d_s.parse().expect("ratio denominator");
            assert!(n > d && d >= 1);
            1_200.0 * ((n as f64) / (d as f64)).log2()
        } else {
            let n: u64 = token.parse().expect("integer ratio");
            1_200.0 * (n as f64).log2()
        };
        assert!(cents.is_finite() && cents > 0.0);
        if let Some(&prev) = cents_out.last() {
            assert!(cents >= prev, "Scala pitches must be non-decreasing");
        }
        cents_out.push(cents);
    }
    assert_eq!(cents_out.len(), count);
    cents_out
}

fn parse_and_validate_kbm(kbm: &str, expected_degrees: usize) {
    let values: Vec<&str> = kbm
        .lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('!') && !l.is_empty())
        .collect();
    assert!(values.len() >= 7 + expected_degrees);
    let map_size: usize = values[0].parse().unwrap();
    let first_midi: u8 = values[1].parse().unwrap();
    let last_midi: u8 = values[2].parse().unwrap();
    let middle_note: u8 = values[3].parse().unwrap();
    let ref_note: u8 = values[4].parse().unwrap();
    let ref_freq: f64 = values[5].parse().unwrap();
    let octave_deg: usize = values[6].parse().unwrap();

    assert_eq!(map_size, expected_degrees);
    assert_eq!((first_midi, last_midi), (0, 127));
    assert!(middle_note <= 127 && ref_note <= 127);
    assert!((20.0..=8_000.0).contains(&ref_freq));
    assert_eq!(octave_deg, expected_degrees);
    for (idx, entry) in values[7..7 + map_size].iter().enumerate() {
        assert_eq!(entry.parse::<usize>().unwrap(), idx);
    }
}

#[test]
fn test_midi_format1_header_vlq_channel9_avoidance_and_prime_pitch_bends() {
    for sample in [0_u32, 1, 127, 128, 480, 16_383, 65_535, 0x0FFF_FFFF] {
        let mut buf = Vec::new();
        encode_vlq(sample, &mut buf);
        let (decoded, used) = decode_vlq(&buf).expect("VLQ must decode");
        assert_eq!((decoded, used), (sample, buf.len()));
    }

    let primes = [2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
    for p in primes {
        let (num, den) = just_intonation_rational(p);
        let hz = f64::from(PRIME_CHORD_ROOT_HZ) * (num as f64) / (den as f64);
        let pitch = MidiPitch::from_freq_hz(hz as f32);
        let err = pitch.cents_error(hz);
        assert!(
            err <= 0.5,
            "prime {p} ({num}/{den}) pitch-bend error {err:.4} cents exceeded 0.5 cents"
        );
    }

    let chord = prime_chord_frame(
        &[(2, 1), (3, 1), (5, 1), (7, 1), (11, 1), (13, 1)],
        WaveformPreset::JustHarmonic,
    );
    let midi = encode_frames_to_midi("Prime Harmonics", &[chord; 12], 1.0 / 60.0);
    assert_eq!(&midi[0..4], b"MThd");
    assert_eq!(u32::from_be_bytes(midi[4..8].try_into().unwrap()), 6);
    assert_eq!(u16::from_be_bytes(midi[8..10].try_into().unwrap()), 1);
    let ntrks = u16::from_be_bytes(midi[10..12].try_into().unwrap());
    assert_eq!(ntrks, 7);
    assert_eq!(
        u16::from_be_bytes(midi[12..14].try_into().unwrap()),
        MIDI_TICKS_PER_QUARTER
    );

    verify_smf_tracks_and_no_percussion_channel(&midi, ntrks as usize);

    let span_midi = encode_midi_voice_tracks(
        "Span Score",
        &[vec![MidiNoteSpan {
            start_tick: 0,
            duration_ticks: seconds_to_ticks(1.0),
            freq_hz: 330.0,
            velocity: 96,
            pan: -0.25,
        }]],
    );
    verify_smf_tracks_and_no_percussion_channel(&span_midi, 2);

    let ramp_frames: Vec<_> = (0..12)
        .map(|i| {
            let mut f = super::frame::SonificationFrame::silent();
            f.active_voices = 1;
            f.master_gain = 1.0;
            f.voices[0].freq_hz = 440.0;
            f.voices[0].gain = 0.25 + (i as f32) * 0.008;
            f.voices[0].pan = -0.2 + (i as f32) * 0.016;
            f
        })
        .collect();
    let ramp_midi = encode_frames_to_midi("Gradual Ramp", &ramp_frames, 1.0 / 60.0);
    let cc11_events = ramp_midi.windows(2).filter(|w| w == &[0xB0, 11]).count();
    let cc10_events = ramp_midi.windows(2).filter(|w| w == &[0xB0, 10]).count();
    assert!(cc11_events >= 2 && cc10_events >= 2);
}

fn verify_smf_tracks_and_no_percussion_channel(midi: &[u8], expected_tracks: usize) {
    let mut pos = 14_usize;
    for _ in 0..expected_tracks {
        assert_eq!(&midi[pos..pos + 4], b"MTrk");
        let len = u32::from_be_bytes(midi[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let end = pos + 8 + len;
        let trk = &midi[pos + 8..end];
        assert_eq!(&trk[trk.len() - 3..], &[0xFF, 0x2F, 0x00]);
        assert!(!trk.windows(2).any(|w| w[0] == 0x99 || w[0] == 0xE9));
        pos = end;
    }
    assert_eq!(pos, midi.len());
}

#[test]
fn test_scala_scl_and_kbm_huygens_fokker_surge_vital_parser() {
    let factors = [(2, 1), (3, 2), (5, 1), (7, 1), (11, 1), (13, 1)];
    let ji_scale = ScalaScale::from_prime_factors("ji_13_limit", "13-Limit Prime Chord", &factors);
    assert!(ji_scale.pitches.contains(&ScalaPitch::Ratio(5, 4)));
    assert!(ji_scale.pitches.contains(&ScalaPitch::Ratio(3, 2)));
    assert!(ji_scale.pitches.contains(&ScalaPitch::Ratio(7, 4)));
    assert_eq!(ji_scale.pitches.last(), Some(&ScalaPitch::Ratio(2, 1)));

    let (scl, kbm) = ji_scale.to_scl_and_kbm_default();
    let cents = parse_and_validate_scl(&scl);
    assert!((cents.last().copied().unwrap() - 1_200.0).abs() < 1e-6);
    parse_and_validate_kbm(&kbm, cents.len());

    let modal = ScalaScale::from_modal_ratios(
        "chladni_leissa",
        "Clamped Square Plate Modes",
        &[2.039_6, 3.007_2, 3.656_5],
    );
    let modal_scl = modal.to_scl_string();
    let modal_cents = parse_and_validate_scl(&modal_scl);
    let modal_kbm = modal.to_kbm_string(60, 114.54);
    parse_and_validate_kbm(&modal_kbm, modal_cents.len());
}

#[test]
fn test_offline_multi_second_wav_44k1_48k_and_zero_alloc() {
    let clamped_low = OfflineRenderConfig {
        sample_rate: 22_050,
        duration_secs: 0.1,
        master_volume: 0.8,
    }
    .sanitized();
    assert_eq!(clamped_low.sample_rate, WAV_EXPORT_SAMPLE_RATE);
    assert_eq!(clamped_low.duration_secs, 1.0);

    let clamped_high = OfflineRenderConfig {
        sample_rate: WAV_STUDIO_SAMPLE_RATE,
        duration_secs: 42.0,
        master_volume: 0.9,
    }
    .sanitized();
    assert_eq!(clamped_high.sample_rate, WAV_STUDIO_SAMPLE_RATE);
    assert_eq!(clamped_high.duration_secs, 15.0);
    assert_eq!(clamped_high.total_stereo_frames(), 720_000);

    let chord = prime_chord_frame(&[(2, 1), (3, 1), (5, 1)], WaveformPreset::JustHarmonic);
    let cfg_48k = OfflineRenderConfig {
        sample_rate: WAV_STUDIO_SAMPLE_RATE,
        duration_secs: 1.25,
        master_volume: 0.8,
    };
    let wav_a = synthesize_offline_wav(cfg_48k, |_step, _t| chord);
    let wav_b = synthesize_offline_wav(cfg_48k, |_step, _t| chord);
    assert_eq!(wav_a, wav_b, "Offline WAV synthesis must be deterministic");
    assert_eq!(
        u32::from_le_bytes(wav_a[24..28].try_into().unwrap()),
        WAV_STUDIO_SAMPLE_RATE
    );
    assert_eq!(wav_a.len(), 44 + 60_000 * 4);

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut synth = OfflinePcmSynthesizer::new(WAV_STUDIO_SAMPLE_RATE, 800);
    let _ = synth.set_muted_from_user_gesture(false);
    let mut pcm = vec![0.0_f32; 48_000 * 2];
    let (frames, _metrics) = VerifiedEngine::run_verified(|| {
        render_offline_pcm_into(&mut synth, 800, &mut pcm, |_step, _t| chord)
    });
    assert_eq!(frames, 48_000);
    assert!(pcm.iter().all(|s| s.abs() <= LIMITER_CEILING_LINEAR + 1e-6));
}
