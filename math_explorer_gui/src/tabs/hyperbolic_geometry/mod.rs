// @explorer_feature = "pure_math"
use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod billiards_tool;
mod billiards_ui;
pub mod disk_tool;
mod disk_ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

/// Synchronizes the canonical URL hash route for [`HyperbolicGeometryTab`] onto `#the_canvas_id`
/// in browser builds and returns the canonical route fragment.
#[cfg(target_arch = "wasm32")]
pub(crate) fn sync_wasm_hyperbolic_route(active_tool_id: Option<&str>) -> String {
    let route = match active_tool_id {
        Some(tool_id) if !tool_id.is_empty() => {
            format!("#tab=hyperbolic_geometry&tool={tool_id}")
        }
        _ => String::from("#tab=hyperbolic_geometry"),
    };
    if let Some(canvas) = web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| doc.get_element_by_id("the_canvas_id"))
    {
        let _ = canvas.set_attribute("data-hyperbolic-route", &route);
    }
    route
}

/// Standalone Pure Math tab for intrinsic $\mathrm{SU}(1,1)$ hyperbolic disk geometry,
/// Fuchsian $\{p, q\}$ tessellations, and chaotic geodesic billiards.
pub struct HyperbolicGeometryTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for HyperbolicGeometryTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("hyperbolic_geometry"),
        }
    }
}

impl ExplorerTab for HyperbolicGeometryTab {
    fn name(&self) -> &'static str {
        "Hyperbolic Geometry"
    }

    fn academic_domain(&self) -> &'static str {
        "Pure Math"
    }

    fn url_slug(&self) -> &'static str {
        "hyperbolic_geometry"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "hyperbolic_geometry");
        #[cfg(target_arch = "wasm32")]
        {
            let _ = sync_wasm_hyperbolic_route(self.active_tool_id());
        }
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework
            .active_tool
            .as_ref()
            .and_then(|tool| tool.focused_canvas_id())
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

// [cite:modular_polynomials_review]
