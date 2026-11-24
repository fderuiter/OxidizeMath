#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::collections::HashMap;

use super::{UnifiedModel, UnifiedSimTool};

impl<M: UnifiedModel> UnifiedSimTool<M> {
    fn build_deterministic_export_model(&self) -> Result<(M, HashMap<String, f64>), String> {
        let params = self
            .params
            .read()
            .map_err(|_| "Failed to read simulation parameters".to_string())?
            .clone();
        let mut model = M::new(&params);
        if self.custom_formula_enabled {
            let payload = self
                .shared_formula
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            model.apply_compiled_formula(payload.as_ref(), &params);
        }
        if M::supports_spatial_brush() {
            let sculpt = self.sculpt_state.read().unwrap_or_else(|e| e.into_inner());
            if !sculpt.is_empty() {
                model.apply_sculpt_payload(&sculpt, &params);
            }
        }
        Ok((model, params))
    }

    pub(super) fn render_unified_wav(
        &self,
        sample_rate: u32,
        duration_secs: f32,
    ) -> Result<Vec<u8>, String> {
        if !M::supports_sonification() {
            return Err(format!(
                "WAV audio export is not supported for {}",
                M::name()
            ));
        }
        let (mut model, params) = self.build_deterministic_export_model()?;
        let substeps = self.steps_per_frame.max(1);
        let pitch_scale = self.audio_settings.pitch_scale;
        Ok(crate::audio::render_frames_wav(
            sample_rate,
            duration_secs,
            |_, _| {
                for _ in 0..substeps {
                    model.step(&params);
                }
                model
                    .sonification_frame()
                    .unwrap_or_else(|| model.get_snapshot().sonification)
                    .with_pitch_scale(pitch_scale)
            },
        ))
    }

    pub(super) fn render_unified_midi(&self, duration_secs: f32) -> Result<Vec<u8>, String> {
        if !M::supports_sonification() {
            return Err(format!(
                "MIDI score export is not supported for {}",
                M::name()
            ));
        }
        let (mut model, params) = self.build_deterministic_export_model()?;
        let substeps = self.steps_per_frame.max(1);
        let pitch_scale = self.audio_settings.pitch_scale;
        Ok(crate::audio::render_frames_midi(
            M::name(),
            duration_secs,
            |_, _| {
                for _ in 0..substeps {
                    model.step(&params);
                }
                model
                    .sonification_frame()
                    .unwrap_or_else(|| model.get_snapshot().sonification)
                    .with_pitch_scale(pitch_scale)
            },
        ))
    }
}
