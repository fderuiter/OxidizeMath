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

    /// Renders the content of the tab.
    ///
    /// The tab is responsible for defining its own panels (e.g., `SidePanel`, `CentralPanel`).
    /// The parent application will typically have already drawn a top navigation bar, so avoid `TopBottomPanel::top`.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The egui Context, used for adding widgets and handling input.
    /// * `frame` - The eframe Frame, used for window management (e.g., resizing, closing).
    fn show(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame);

    /// Serializes the tab's parameter and panel state into a String.
    fn save_state(&self) -> Option<String> {
        None
    }

    /// Deserializes and restores the tab's parameter and panel state from a String.
    fn load_state(&mut self, _state: &str) {}
}
