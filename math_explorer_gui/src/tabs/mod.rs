#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

include!(concat!(env!("OUT_DIR"), "/generated_tabs.rs"));

/// A trait for defining a tab in the Math Explorer application.
///
/// This trait serves as the primary extension point for the GUI. Each mathematical domain
/// (e.g., MRI Physics, Game Theory) should implement this trait as a standalone struct.
///
/// # Automated Plugin Discovery
///
/// The `math_explorer_gui` application uses an automated build-time plugin discovery mechanism.
/// To add a new tab to the GUI, you simply need to create a new module file in the `src/tabs` directory
/// (or in any workspace dependency), and implement the `ExplorerTab` trait for your struct.
///
/// You MUST NOT manually edit `app.rs` or `tabs/mod.rs` to register your new tab. The `build.rs` script
/// will automatically scan for the `impl ExplorerTab for YourStruct` pattern and generate the necessary
/// inclusion and instantiation logic.
///
/// ## Configuration Attributes
///
/// You can configure how your tab is built and displayed by adding special magic comments at the top of your module file:
///
/// - `// @explorer_feature = "feature_name"`: Tells the build script that this tab should only be included if the specified Cargo feature is enabled.
/// - `// @explorer_order = 10`: Defines the order in which this tab appears in the navigation bar. Lower numbers appear first.
///
/// # Example
///
/// ```rust,no_run
/// // @explorer_feature = "pure_math"
/// // @explorer_order = 1
///
/// use math_explorer_gui::tabs::ExplorerTab;
/// use eframe::egui;
///
/// #[derive(Default)]
/// pub struct MyTab {
///     counter: i32,
/// }
///
/// impl ExplorerTab for MyTab {
///     fn name(&self) -> &'static str {
///         "My Tab"
///     }
///
///     fn show(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
///         egui::CentralPanel::default().show(ctx, |ui| {
///             ui.heading("My Tab");
///             if ui.button("Increment").clicked() {
///                 self.counter += 1;
///             }
///             ui.label(format!("Count: {}", self.counter));
///         });
///     }
/// }
/// ```
pub trait ExplorerTab {
    /// Returns the name of the tab, displayed in the navigation bar.
    fn name(&self) -> &'static str;

    /// Renders the content of the tab at the root `egui::Context` level.
    ///
    /// By default, this creates a frameless root `CentralPanel` and delegates to [`Self::show_inside`].
    fn show(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                self.show_inside(ui, frame);
            });
    }

    /// Renders the content of the tab inside a scoped parent `egui::Ui` (such as a multi-pane viewport region).
    fn show_inside(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.show(ui.ctx(), frame);
    }

    /// Monotonic or deterministic revision token for persisted tab state (`0` if untracked).
    fn state_revision(&self) -> u64 {
        0
    }

    /// Serializes the tab's parameter and panel state into a String.
    fn save_state(&self) -> Option<String> {
        None
    }

    /// Deserializes and restores the tab's parameter and panel state from a String.
    fn load_state(&mut self, _state: &str) {}

    /// Academic domain category of the tab for top navigation bar grouping.
    fn academic_domain(&self) -> &'static str {
        match self.name() {
            "Analysis & Calculus"
            | "Financial Math"
            | "Geometry & Topology"
            | "Graph Theory"
            | "Number Theory" => "Pure Math",

            "Chaos Theory"
            | "Fluid Dynamics"
            | "Medical Physics"
            | "MRI Physics"
            | "Quantum Mechanics"
            | "Solid State Physics" => "Physics",

            "Battery Degradation"
            | "Clinical Trials"
            | "Favoritism"
            | "Game Theory"
            | "Traceability Portal" => "Applied Math",

            "Morphogenesis" | "Neuroscience" | "Epidemiology" => "Biology",

            "Climate" => "Climate",

            "Artificial Intelligence" => "AI",

            _ => "Applied Math",
        }
    }

    /// Activates a tool within this tab by name.
    /// Returns true if a matching tool was found and activated.
    fn select_tool_by_name(&mut self, _name: &str) -> bool {
        false
    }

    /// Returns a reference to the underlying `SimulationFramework` in test builds.
    #[cfg(test)]
    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        None
    }

    /// Returns a mutable reference to the underlying `SimulationFramework` in test builds.
    #[cfg(test)]
    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        None
    }
}
