#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::framework::InteractiveTool;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn step_unified_model(&mut self) {
        if let Ok(params) = self.params.read() {
            let mut model = M::new(&params);
            if self.custom_formula_enabled {
                let payload = self
                    .shared_formula
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                model.apply_compiled_formula(payload.as_ref(), &params);
            }
            for _ in 0..self.steps_per_frame.max(1) {
                model.step(&params);
            }
            self.stepped_snapshot = Some(model.get_snapshot());
        }
    }

    pub(super) fn compute_unified_state_revision(&self) -> u64 {
        let params = self.params.read().unwrap_or_else(|e| e.into_inner());
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for (key, _) in &self.param_metadata {
            key.hash(&mut h);
            if let Some(val) = params.get(key) {
                val.to_bits().hash(&mut h);
            }
        }
        if M::supports_sonification() {
            self.audio_settings.gain.to_bits().hash(&mut h);
            self.audio_settings.pitch_scale.to_bits().hash(&mut h);
        }
        if let Some(ext) = &self.ui_extension {
            ext.state_revision_salt().hash(&mut h);
        }
        if let Some(atlas) = &self.atlas {
            atlas.is_visible().hash(&mut h);
            atlas.axis_keys().hash(&mut h);
        }
        self.custom_formula_enabled.hash(&mut h);
        if let Some(ref bar) = self.formula_bar {
            bar.raw_text.hash(&mut h);
            for &pv in bar.param_slice() {
                pv.to_bits().hash(&mut h);
            }
        }
        h.finish().max(1)
    }

    pub(super) fn current_sonification_frame(&self) -> Option<crate::audio::SonificationFrame> {
        if !M::supports_sonification() {
            return None;
        }
        let snapshot = self
            .stepped_snapshot
            .as_ref()
            .or_else(|| self.controller.latest_snapshot())?;
        let active = (self.stepped_snapshot.is_some() || self.controller.state().is_running())
            && self.has_finite_parameters();
        if active {
            Some(snapshot.sonification.sanitize())
        } else {
            Some(crate::audio::SonificationFrame::silent())
        }
    }

    pub(super) fn sample_model_sonification_frames(
        &mut self,
        num_frames: usize,
    ) -> Vec<crate::audio::SonificationFrame> {
        if !M::supports_sonification() {
            return Vec::new();
        }
        let count = num_frames.max(1);
        let params = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
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
        let sub_steps = self.steps_per_frame.clamp(1, 20);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            for _ in 0..sub_steps {
                model.step(&params);
            }
            let frame = model
                .sonification_frame()
                .unwrap_or_else(|| model.get_snapshot().sonification);
            out.push(frame.sanitize());
        }
        out
    }
}

impl<M: UnifiedModel> scientific_metadata::theory::TheoryDescribable for UnifiedSimTool<M> {
    fn theory_description(&self) -> String {
        self.cached_theory_desc.clone()
    }
    fn theory_latex_equation(&self) -> Option<String> {
        self.cached_latex_eq.clone()
    }
    fn phonetic_description(&self) -> String {
        self.cached_phonetic.clone()
    }
    fn theory_citation(&self) -> String {
        self.cached_citation.clone()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        self.cached_descs.clone()
    }
    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        self.param_metadata.iter().cloned().collect()
    }
}
