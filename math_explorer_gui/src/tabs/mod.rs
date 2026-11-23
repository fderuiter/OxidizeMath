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

    /// Returns the active tool's interactive canvas `Id` when focus is owned by a non-text canvas.
    fn focused_canvas_id(&self) -> Option<egui::Id> {
        None
    }

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

    /// Returns a reference to the underlying [`crate::framework::SimulationFramework`] when backed by one.
    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        None
    }

    /// Returns a mutable reference to the underlying [`crate::framework::SimulationFramework`] when backed by one.
    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        None
    }

    /// Applies the root-hoisted `StarterGallery` modal outcome to this tab instance.
    fn apply_starter_gallery_outcome(
        &mut self,
        ctx: &egui::Context,
        id_source: &str,
        is_open: bool,
        selected: Option<crate::starter_gallery::StarterCard>,
    ) {
        if let Some(fw) = self.framework_mut() {
            fw.apply_starter_gallery_outcome(ctx, id_source, is_open, selected);
        }
    }

    /// Applies either a starter card or inverse challenge selection from `StarterGallery`.
    fn apply_starter_gallery_selection(
        &mut self,
        ctx: &egui::Context,
        id_source: &str,
        is_open: bool,
        selected: Option<crate::starter_gallery::GallerySelection>,
    ) {
        match selected {
            Some(crate::starter_gallery::GallerySelection::Starter(card)) => {
                self.apply_starter_gallery_outcome(ctx, id_source, is_open, Some(card));
            }
            other => {
                if let Some(fw) = self.framework_mut() {
                    fw.apply_starter_gallery_selection(ctx, id_source, is_open, other);
                }
            }
        }
    }

    /// Synchronizes this tab's active tool with [`crate::app::workspace_bus::SharedWorkspaceBus`] when linked.
    fn sync_workspace_bus(&mut self, bus: &mut crate::app::workspace_bus::SharedWorkspaceBus) {
        if let Some(fw) = self.framework_mut() {
            if let Some(tool) = fw.active_tool.as_mut() {
                tool.sync_workspace_bus(bus);
            }
        }
    }

    /// Canonical URL slug identifying this tab in `#tab=<slug>` share links.
    fn url_slug(&self) -> &'static str {
        match self.name() {
            "Artificial Intelligence" => "ai",
            "Analysis & Calculus" => "analysis",
            "Battery Degradation" => "battery_degradation",
            "Chaos Theory" => "chaos",
            "Climate" => "climate",
            "Clinical Trials" => "clinical_trials",
            "Epidemiology" => "epidemiology",
            "Favoritism" => "favoritism",
            "Financial Math" => "financial_math",
            "Fluid Dynamics" => "fluid_dynamics",
            "Game Theory" => "game_theory",
            "Geometry & Topology" => "geometry_topology",
            "Graph Theory" => "graph_theory",
            "Medical Physics" => "medical",
            "Morphogenesis" => "morphogenesis",
            "MRI Physics" => "mri",
            "Neuroscience" => "neuroscience",
            "Number Theory" => "number_theory",
            "Quantum Mechanics" => "quantum",
            "Solid State Physics" => "solid_state",
            "Traceability Portal" => "traceability",
            _ => "simulation",
        }
    }

    /// Active `InteractiveTool` identifier for `#tool=<id>` share links (`None` for single-view tabs).
    fn active_tool_id(&self) -> Option<&'static str> {
        self.framework().and_then(|fw| fw.active_tool_id())
    }

    /// Active inverse `ChallengeMetadata` identifier for `#challenge=<id>` share links.
    fn active_challenge_id(&self) -> Option<&'static str> {
        self.framework().and_then(|fw| fw.active_challenge_id())
    }

    /// Deterministic parameter-only revision token for 300 ms debounced URL sync.
    fn url_state_revision(&self) -> u64 {
        self.framework()
            .map_or_else(|| self.state_revision(), |fw| fw.url_state_revision())
    }

    /// Exports active simulation parameters as a compact `PresetWrapper` JSON envelope for URL sharing.
    fn export_url_preset_json(&self) -> Result<Option<String>, String> {
        match self.framework() {
            Some(fw) => fw.export_url_preset_json(),
            None => Ok(None),
        }
    }

    /// Validates and applies a shared URL tool identifier and `PresetWrapper` JSON payload.
    fn import_url_preset_json(
        &mut self,
        tool_id: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        if let Some(fw) = self.framework_mut() {
            return fw.import_url_preset_json(tool_id, preset_json);
        }
        if let Some(unexpected_tool) = tool_id {
            return Err(format!(
                "Tab '{}' does not support sub-tool '{unexpected_tool}'",
                self.name()
            ));
        }
        if preset_json.is_some() {
            return Err(format!(
                "URL parameter state import is not supported for '{}'",
                self.name()
            ));
        }
        Ok(())
    }

    /// Validates and applies shared URL `tool`, `challenge`, and `state` parameters.
    fn import_url_share_params(
        &mut self,
        tool_id: Option<&str>,
        challenge_id: Option<&str>,
        preset_json: Option<&str>,
    ) -> Result<(), String> {
        if challenge_id.is_none() {
            return self.import_url_preset_json(tool_id, preset_json);
        }
        if let Some(fw) = self.framework_mut() {
            return fw.import_url_share_params(tool_id, challenge_id, preset_json);
        }
        Err(format!(
            "Tab '{}' does not support inverse challenges",
            self.name()
        ))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::path::Path;

    fn read_build_script_output(out_dir: &Path) -> String {
        let parent_dir = out_dir
            .parent()
            .expect("OUT_DIR must have a parent build directory");
        let build_output_path = [
            parent_dir.join("output"),
            parent_dir.join("run").join("stdout"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .expect("Cargo build script output file must exist");
        fs::read_to_string(&build_output_path)
            .expect("Cargo build script output file must be readable")
    }

    #[test]
    fn test_build_script_emits_rerun_directives_for_tabs_and_path_deps() {
        let out_dir = Path::new(env!("OUT_DIR"));
        let generated_tabs = fs::read_to_string(out_dir.join("generated_tabs.rs"))
            .expect("OUT_DIR/generated_tabs.rs must be generated by build.rs");
        assert!(
            generated_tabs.contains("pub fn instantiate_tabs()"),
            "generated_tabs.rs should define instantiate_tabs"
        );

        let build_output = read_build_script_output(out_dir);
        let rerun_paths: HashSet<&str> = build_output
            .lines()
            .filter_map(|line| line.strip_prefix("cargo:rerun-if-changed="))
            .collect();

        for path in &rerun_paths {
            assert!(!path.trim().is_empty(), "rerun directive must not be empty");
            assert!(
                !path.contains('\\'),
                "rerun path must use forward slashes: {path}"
            );
        }

        for expected in [
            "build.rs",
            "src/tabs",
            "Cargo.toml",
            "../crates/federated_registry/src",
            "../crates/oxidize_core/src",
            "../math_explorer/src",
            "../crates/math_commons/src",
            "../crates/scientific_metadata/src",
            "../crates/diagnostics/src",
            "../crates/federated_registry/src/lib.rs",
            "../crates/oxidize_core/src/lib.rs",
            "../math_explorer/src/lib.rs",
            "../crates/math_commons/src/lib.rs",
            "../crates/math_commons/src/ast",
            "../crates/scientific_metadata/src/lib.rs",
            "../crates/diagnostics/src/lib.rs",
        ] {
            assert!(
                rerun_paths.contains(expected),
                "Missing expected cargo:rerun-if-changed={expected} in math_explorer_gui output"
            );
        }
    }
}
