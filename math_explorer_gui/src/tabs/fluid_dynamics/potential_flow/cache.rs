#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    ElementType, PotentialFlowTool, SerializedFlowElement, FLOW_ANGLE_MAX, FLOW_ANGLE_MIN,
    FLOW_POS_MAX, FLOW_POS_MIN, FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN, FLOW_VIEW_MAX, FLOW_VIEW_MIN,
};
use egui_plot::PlotPoint;
use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

const fn default_element_strength() -> f64 {
    10.0
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PotentialFlowState {
    elements: Vec<SerializedFlowElement>,
    grid_size: usize,
    view_range: f64,
    #[serde(default)]
    selected_element_type: ElementType,
    #[serde(default = "default_element_strength")]
    new_element_strength: f64,
    #[serde(default)]
    new_element_x: f64,
    #[serde(default)]
    new_element_y: f64,
    #[serde(default)]
    new_element_angle: f64,
}

impl PotentialFlowState {
    fn is_valid(&self) -> bool {
        (2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
            && self.view_range.is_finite()
            && (FLOW_VIEW_MIN..=FLOW_VIEW_MAX).contains(&self.view_range)
            && self.new_element_strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.new_element_strength)
            && self.new_element_x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_x)
            && self.new_element_y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_y)
            && self.new_element_angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.new_element_angle)
            && self.elements.len() <= crate::framework::MAX_JSON_ARRAY_LEN
            && self.elements.iter().all(SerializedFlowElement::is_valid)
    }
}

#[derive(Default)]
pub(crate) struct ArrowFieldCache {
    key: Option<u64>,
    origins: Arc<Vec<PlotPoint>>,
    tips: Arc<Vec<PlotPoint>>,
    pub(crate) eval_count: usize,
}

fn canonical_bits(val: f64) -> u64 {
    if val == 0.0 {
        0
    } else {
        val.to_bits()
    }
}

fn hash_flow_element(el: &SerializedFlowElement, h: &mut impl std::hash::Hasher) {
    (el.element_type as u8).hash(h);
    canonical_bits(el.strength).hash(h);
    canonical_bits(el.x).hash(h);
    canonical_bits(el.y).hash(h);
    canonical_bits(el.angle).hash(h);
}

impl PotentialFlowTool {
    pub(crate) fn save_state_json(&self) -> Option<String> {
        serde_json::to_string(&PotentialFlowState {
            elements: self.elements.clone(),
            grid_size: self.grid_size,
            view_range: self.view_range,
            selected_element_type: self.selected_element_type,
            new_element_strength: self.new_element_strength,
            new_element_x: self.new_element_x,
            new_element_y: self.new_element_y,
            new_element_angle: self.new_element_angle,
        })
        .ok()
    }

    pub(crate) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let loaded: PotentialFlowState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid PotentialFlowTool state: {e}"))?;
        if !loaded.is_valid() {
            return Err(
                "PotentialFlowTool parameters must be finite and within valid grid/view bounds"
                    .to_string(),
            );
        }
        self.elements = loaded.elements;
        self.grid_size = loaded
            .grid_size
            .min(crate::presets::MAX_INTERACTIVE_GRID_SIZE);
        self.view_range = loaded.view_range;
        self.selected_element_type = loaded.selected_element_type;
        self.new_element_strength = loaded.new_element_strength;
        self.new_element_x = loaded.new_element_x;
        self.new_element_y = loaded.new_element_y;
        self.new_element_angle = loaded.new_element_angle;
        self.rebuild_field();
        Ok(())
    }

    pub(crate) fn field_geometry_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.grid_size.hash(&mut h);
        canonical_bits(self.view_range).hash(&mut h);
        self.elements.len().hash(&mut h);
        for el in &self.elements {
            hash_flow_element(el, &mut h);
        }
        h.finish().max(1)
    }

    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.field_geometry_key().hash(&mut h);
        (self.selected_element_type as u8).hash(&mut h);
        canonical_bits(self.new_element_strength).hash(&mut h);
        canonical_bits(self.new_element_x).hash(&mut h);
        canonical_bits(self.new_element_y).hash(&mut h);
        canonical_bits(self.new_element_angle).hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn ensure_cached_arrows(&mut self) -> (Arc<Vec<PlotPoint>>, Arc<Vec<PlotPoint>>) {
        let key = self.field_geometry_key();
        if self.cache.key != Some(key) {
            let (raw_origins, raw_tips) = self.compute_arrow_field();
            let origins = raw_origins
                .into_iter()
                .map(|[x, y]| PlotPoint::new(x, y))
                .collect();
            let tips = raw_tips
                .into_iter()
                .map(|[x, y]| PlotPoint::new(x, y))
                .collect();
            self.cache.origins = Arc::new(origins);
            self.cache.tips = Arc::new(tips);
            self.cache.key = Some(key);
            self.cache.eval_count += 1;
        }
        (
            Arc::clone(&self.cache.origins),
            Arc::clone(&self.cache.tips),
        )
    }
}
