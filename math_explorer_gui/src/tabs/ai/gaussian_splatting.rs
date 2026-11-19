use crate::framework::InteractiveTool;
use eframe::egui;

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct GaussianSplattingTool {
    pub scale: f64,
}

impl Default for GaussianSplattingTool {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GaussianSplattingState {
    scale: f64,
}

impl InteractiveTool for GaussianSplattingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Gaussian Splatting"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }
    fn domain(&self) -> &'static str {
        "ai"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&GaussianSplattingState { scale: self.scale }).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: GaussianSplattingState =
            serde_json::from_str(state).map_err(|e| format!("Invalid splatting state: {e}"))?;
        if !parsed.scale.is_finite() || !(0.1..=10.0).contains(&parsed.scale) {
            return Err("Gaussian splat scale must be finite within [0.1, 10.0]".into());
        }
        self.scale = parsed.scale;
        Ok(())
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("gaussian_splatting_controls").show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();
            ui.add(egui::Slider::new(&mut self.scale, 0.1..=10.0).text("Scale"));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("3D Gaussian Splatting Visualization");
            ui.label("Placeholder for Gaussian Splatting Viewer.");
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for GaussianSplattingTool {
    fn theory_description(&self) -> String {
        "3D Gaussian Splatting is a rasterization-based technique for real-time rendering of radiance fields, using explicit 3D Gaussians to represent scenes instead of implicit neural representations.".into()
    }
    fn phonetic_description(&self) -> String {
        "3D Gaussian Splatting is a rasterization-based technique for real-time rendering of radiance fields, using explicit 3D Gaussians to represent scenes instead of implicit neural representations.".into()
    }
    fn theory_citation(&self) -> String {
        "Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics.".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "scale".into(),
            "The scale of the 3D Gaussians representing the scene geometry.".into(),
        );
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "Gaussian Splatting",
        display_name: "Gaussian Splatting",
        domain: "ai",
        tags: &["ai", "graphics"],
        build: || Box::new(GaussianSplattingTool::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gaussian Splatting",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "scale": 1.5
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gaussian_splatting_starter",
        title: "Gaussian Splatting",
        description: "3D Gaussian radiance field scale parameter control and theoretical model overview.",
        category: "Neural Rendering",
        icon: "🎨",
        domain: "ai",
        tool_name: Some("Gaussian Splatting"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
