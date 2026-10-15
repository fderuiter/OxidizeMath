use crate::framework::InteractiveTool;
use eframe::egui;

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
#[derive(Default)]
pub struct GaussianSplattingTool {
    pub scale: f64,
}

impl InteractiveTool for GaussianSplattingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Gaussian Splatting"
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
        map.insert("scale".into(), "The scale of the 3D Gaussians representing the scene geometry.".into());
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "Gaussian Splatting",
        domain: "ai",
        tags: &["ai", "graphics"],
        build: || Box::new(GaussianSplattingTool::default()),
    }
}
