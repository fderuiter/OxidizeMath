#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{UnifiedModel, UnifiedSimTool};
use crate::framework::InteractiveTool;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

impl<M: UnifiedModel> UnifiedSimTool<M> {
    pub(super) fn current_sonification_frame(&self) -> Option<crate::audio::SonificationFrame> {
        if !M::supports_sonification() {
            return None;
        }
        let snapshot = self
            .stepped_snapshot
            .as_ref()
            .or_else(|| self.controller.latest_snapshot())?;
        let mut frame = M::sonification_from_snapshot(snapshot)?;
        frame.active = self.controller.state().is_running() && self.has_finite_parameters();
        Some(frame.sanitized())
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
            if let Some(mut frame) = model.sonification_frame() {
                frame.active = true;
                out.push(frame.sanitized());
            }
        }
        out
    }
}

impl<M: UnifiedModel> scientific_metadata::theory::TheoryDescribable for UnifiedSimTool<M> {
    fn theory_description(&self) -> String {
        self.cached_theory_desc.clone()
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
