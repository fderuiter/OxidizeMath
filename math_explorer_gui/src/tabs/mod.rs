#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

#[cfg(feature = "ai")]
pub mod ai;
#[cfg(feature = "pure_math")]
pub mod analysis;
#[cfg(feature = "physics")]
pub mod astrophysics;
#[cfg(feature = "applied")]
pub mod battery_degradation;
#[cfg(feature = "physics")]
pub mod chaos;
#[cfg(feature = "climate")]
pub mod climate;
#[cfg(feature = "applied")]
pub mod clinical_trials;
#[cfg(feature = "epidemiology")]
pub mod epidemiology;
#[cfg(feature = "applied")]
pub mod favoritism;
#[cfg(feature = "pure_math")]
pub mod financial_math;
#[cfg(feature = "physics")]
pub mod fluid_dynamics;
#[cfg(feature = "applied")]
pub mod game_theory;
#[cfg(feature = "pure_math")]
pub mod geometry_topology;
#[cfg(feature = "pure_math")]
pub mod graph_theory;
#[cfg(feature = "physics")]
pub mod medical;
#[cfg(feature = "biology")]
pub mod morphogenesis;
#[cfg(feature = "physics")]
pub mod mri;
#[cfg(feature = "biology")]
pub mod neuroscience;
#[cfg(feature = "pure_math")]
pub mod number_theory;
#[cfg(feature = "physics")]
pub mod quantum;
#[cfg(feature = "physics")]
pub mod solid_state;
pub mod traceability;

/// Static registration descriptor for a GUI domain tab collected via [`inventory`].
#[derive(Clone, Copy)]
pub struct TabDescriptor {
    /// Canonical module and URL slug used as a deterministic secondary sort key.
    pub id: &'static str,
    /// Primary navigation bar sort priority (lower numbers appear first; default `100`).
    pub order: i32,
    /// Constructor returning a boxed default instance of the tab.
    pub build: fn() -> Box<dyn ExplorerTab>,
}

inventory::collect!(TabDescriptor);

fn sorted_tab_descriptors() -> Vec<&'static TabDescriptor> {
    let mut descriptors: Vec<&'static TabDescriptor> =
        inventory::iter::<TabDescriptor>.into_iter().collect();
    descriptors.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.id.cmp(b.id)));
    descriptors
}

/// Instantiates all enabled GUI tabs in deterministic `(order, id)` navigation order.
pub fn instantiate_tabs() -> Vec<Box<dyn ExplorerTab>> {
    sorted_tab_descriptors()
        .into_iter()
        .map(|d| (d.build)())
        .collect()
}

/// Instantiates a single enabled GUI tab by its deterministic `(order, id)` index.
pub fn instantiate_tab(index: usize) -> Option<Box<dyn ExplorerTab>> {
    sorted_tab_descriptors().get(index).map(|d| (d.build)())
}

/// A trait for defining a tab in the Math Explorer application.
///
/// This trait serves as the primary extension point for the GUI. Each mathematical domain
/// (e.g., MRI Physics, Game Theory) implements this trait as a standalone struct.
///
/// # Static Inventory Registration
///
/// To add a new tab to `math_explorer_gui`:
/// 1. Declare a feature-gated `pub mod your_tab;` item in `src/tabs/mod.rs`.
/// 2. Implement [`ExplorerTab`] and [`Default`] on your tab struct.
/// 3. Register a [`TabDescriptor`] via `inventory::submit!`.
///
/// # Example
///
/// ```rust,ignore
/// use crate::tabs::{ExplorerTab, TabDescriptor};
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
///     fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
///         egui::CentralPanel::default().show_inside(ui, |ui| {
///             ui.heading("My Tab");
///             if ui.button("Increment").clicked() {
///                 self.counter += 1;
///             }
///             ui.label(format!("Count: {}", self.counter));
///         });
///     }
/// }
///
/// inventory::submit! {
///     TabDescriptor {
///         id: "my_tab",
///         order: 100,
///         build: || Box::new(MyTab::default()),
///     }
/// }
/// ```
pub trait ExplorerTab {
    /// Returns the name of the tab, displayed in the navigation bar.
    fn name(&self) -> &'static str;

    /// Renders the content of the tab inside a scoped parent `egui::Ui` (such as a multi-pane viewport region).
    fn show_inside(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame);

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

            "Astrophysics"
            | "Chaos Theory"
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

    /// Canonical URL slug identifying this tab in `#tab=<slug>` share links.
    fn url_slug(&self) -> &'static str {
        match self.name() {
            "Artificial Intelligence" => "ai",
            "Analysis & Calculus" => "analysis",
            "Astrophysics" => "astrophysics",
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
    use super::{instantiate_tab, instantiate_tabs};
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
    fn test_inventory_tabs_preserve_deterministic_slug_order() {
        let tabs = instantiate_tabs();
        assert!(!tabs.is_empty(), "Traceability tab is always enabled");
        let slugs: Vec<&str> = tabs.iter().map(|t| t.url_slug()).collect();
        let mut sorted = slugs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(slugs, sorted, "Tab slugs must be unique and sorted");

        for (idx, tab) in tabs.iter().enumerate() {
            let single = instantiate_tab(idx).expect("instantiate_tab must resolve valid index");
            assert_eq!(single.name(), tab.name());
            assert_eq!(single.url_slug(), tab.url_slug());
        }
        assert!(instantiate_tab(tabs.len()).is_none());

        #[cfg(all(
            feature = "pure_math",
            feature = "applied",
            feature = "ai",
            feature = "biology",
            feature = "climate",
            feature = "epidemiology",
            feature = "physics"
        ))]
        assert_eq!(tabs.len(), 22, "All 22 GUI domain tabs must register");
    }

    #[test]
    fn test_build_script_emits_scoped_rerun_directives() {
        let out_dir = Path::new(env!("OUT_DIR"));
        assert!(
            !out_dir.join("generated_tabs.rs").exists(),
            "OUT_DIR/generated_tabs.rs must no longer be generated"
        );
        let manifest = fs::read_to_string(out_dir.join("traceability_manifest.json"))
            .expect("OUT_DIR/traceability_manifest.json must be generated by build.rs");
        assert!(manifest.contains("\"paper_coverage\""));

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
            "build_snapshot.rs",
            "build_snapshot_scan.rs",
            "src/tabs/mod.rs",
            "../traceability.toml",
            "../papers",
            "../crates/federated_registry/src",
            "../crates/oxidize_core/src",
            "../crates/math_commons/src",
            "../crates/scientific_metadata/src",
            "../crates/diagnostics/src",
        ] {
            assert!(
                rerun_paths.contains(expected),
                "Missing expected cargo:rerun-if-changed={expected} in math_explorer_gui output"
            );
        }

        for excluded in [
            "Cargo.toml",
            "../crates/unified_verification/src",
            "../crates/markdown_tests/src",
            "../crates/verified_engine/src",
            "../crates/verified_engine_macros/src",
            "../apps/xtask/src",
        ] {
            assert!(
                !rerun_paths.contains(excluded),
                "Unexpected cross-workspace rerun directive {excluded} in math_explorer_gui output"
            );
        }
    }
}
