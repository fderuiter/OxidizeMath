//! Zero-allocation FRACTRAN prime-register polyphonic sonification and microtonal studio exports.

use super::{FractranMachineTool, MachineStopReason};
use crate::audio::frame::{SonificationFrame, WaveformPreset, MAX_VOICES};
use crate::audio::mappings::prime_chord_frame;
use crate::audio::scala::ScalaScale;
use crate::audio::synth::OfflinePcmSynthesizer;
use crate::audio::wav::{encode_wav_stereo_pcm16, render_offline_pcm_into};
use crate::audio::SonificationBackend;
use math_explorer::pure_math::algorithmic_information::fractran::FractranCompiledVm;

/// Builds a [`SonificationFrame`] from `registers` into pre-allocated `scratch_factors`
/// without any dynamic heap allocation.
pub(crate) fn build_fractran_frame_into(
    vm: &FractranCompiledVm,
    registers: &[u32],
    fired_rule: Option<usize>,
    hit_pow2: bool,
    scratch_factors: &mut [(u64, u32)],
) -> SonificationFrame {
    let cap = scratch_factors.len().min(MAX_VOICES);
    let mut count = 0_usize;
    for (&prime, &exp) in vm.primes().iter().zip(registers) {
        if exp > 0 && count < cap {
            scratch_factors[count] = (prime, exp);
            count += 1;
        }
    }
    if count == 0 && cap > 0 {
        let first_p = vm.primes().first().copied().unwrap_or(2);
        scratch_factors[0] = (first_p, 1);
        count = 1;
    }
    let mut frame = prime_chord_frame(&scratch_factors[..count], WaveformPreset::JustHarmonic);
    if hit_pow2 && frame.active_voices > 0 {
        frame.voices[0].transient_impulse = 0.85;
    } else if fired_rule.is_some() && frame.active_voices > 0 {
        frame.voices[0].transient_impulse = 0.28;
    }
    frame.sanitize()
}

/// Steps `registers` through `vm` while synthesizing stereo PCM into `out_interleaved`
/// with **zero dynamic heap allocations** inside the loop.
pub(crate) fn render_fractran_pcm_into(
    vm: &FractranCompiledVm,
    registers: &mut [u32],
    scratch_factors: &mut [(u64, u32)],
    synth: &mut OfflinePcmSynthesizer,
    step_frames: usize,
    vm_steps_per_tick: usize,
    out_interleaved: &mut [f32],
) -> usize {
    let steps_each = vm_steps_per_tick.clamp(1, 64);
    render_offline_pcm_into(synth, step_frames, out_interleaved, |step_idx, _t_sec| {
        let mut fired = None;
        let mut pow2 = false;
        if step_idx > 0 && step_idx % 4 == 0 {
            for _ in 0..steps_each {
                match vm.step_registers(registers) {
                    Ok(Some(rule_idx)) => {
                        fired = Some(rule_idx);
                        if vm.power_of_two_exponent(registers).is_some() {
                            pow2 = true;
                        }
                    }
                    Ok(None) | Err(_) => break,
                }
            }
        }
        build_fractran_frame_into(vm, registers, fired, pow2, scratch_factors)
    })
}

impl FractranMachineTool {
    pub(super) fn current_sonification_frame(&self) -> SonificationFrame {
        let mut scratch = [(0_u64, 0_u32); MAX_VOICES];
        let hit_pow2 = matches!(self.stop_reason, MachineStopReason::HitPowerOfTwo(_));
        build_fractran_frame_into(
            &self.vm,
            &self.registers,
            self.last_fired_rule,
            hit_pow2,
            &mut scratch,
        )
    }

    fn export_start_registers(&self) -> Vec<u32> {
        let any_nonzero = self.registers.iter().any(|&e| e > 0);
        if any_nonzero && self.stop_reason != MachineStopReason::Halted {
            self.registers.clone()
        } else {
            self.vm
                .encode_value(self.initial_value.max(2))
                .unwrap_or_else(|_| self.registers.clone())
        }
    }

    pub(super) fn export_fractran_wav(
        &self,
        sample_rate: u32,
        duration_secs: f32,
    ) -> Result<Vec<u8>, String> {
        let base_cfg = crate::audio::current_offline_config();
        let cfg = crate::audio::OfflineRenderConfig {
            sample_rate,
            duration_secs,
            master_volume: base_cfg.master_volume,
        }
        .sanitized();
        let total_frames = cfg.total_stereo_frames();
        let step_frames = cfg.step_stereo_frames();
        let mut synth = OfflinePcmSynthesizer::new(cfg.sample_rate, step_frames);
        synth.set_master_volume(cfg.master_volume);
        let _ = synth.set_muted_from_user_gesture(false);

        let mut regs = self.export_start_registers();
        let mut scratch = [(0_u64, 0_u32); MAX_VOICES];
        let mut stereo = vec![0.0_f32; total_frames * 2];
        let vm_steps = self.steps_per_frame.clamp(1, 16);
        render_fractran_pcm_into(
            &self.vm,
            &mut regs,
            &mut scratch,
            &mut synth,
            step_frames,
            vm_steps,
            &mut stereo,
        );
        Ok(encode_wav_stereo_pcm16(cfg.sample_rate, &stereo))
    }

    pub(super) fn export_fractran_midi(&self, duration_secs: f32) -> Result<Vec<u8>, String> {
        let mut regs = self.export_start_registers();
        let mut scratch = [(0_u64, 0_u32); MAX_VOICES];
        let vm_steps = self.steps_per_frame.clamp(1, 16);
        let title = format!("FRACTRAN {}", self.preset.label());
        Ok(crate::audio::render_frames_midi(
            &title,
            duration_secs,
            |step_idx, _t_sec| {
                let mut fired = None;
                let mut pow2 = false;
                if step_idx > 0 && step_idx % 4 == 0 {
                    for _ in 0..vm_steps {
                        match self.vm.step_registers(&mut regs) {
                            Ok(Some(rule_idx)) => {
                                fired = Some(rule_idx);
                                pow2 |= self.vm.power_of_two_exponent(&regs).is_some();
                            }
                            Ok(None) | Err(_) => break,
                        }
                    }
                }
                build_fractran_frame_into(&self.vm, &regs, fired, pow2, &mut scratch)
            },
        ))
    }

    pub(super) fn export_fractran_scala(&self) -> Result<(String, String), String> {
        let factors: Vec<(u64, u32)> = self.vm.primes().iter().map(|&p| (p, 1)).collect();
        let desc = format!(
            "FRACTRAN {} prime register basis ({} primes)",
            self.preset.label(),
            factors.len()
        );
        let scale = ScalaScale::from_prime_factors("fractran_prime_basis", &desc, &factors);
        Ok(scale.to_scl_and_kbm_default())
    }
}

// [cite:algorithmic_information_rust]
