#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

#[cfg(feature = "ai")]
pub mod ai;
#[cfg(feature = "pure_math")]
pub mod algorithmic_information;
#[cfg(feature = "pure_math")]
pub mod analysis;
#[cfg(feature = "physics")]
pub mod astrophysics;
#[cfg(feature = "applied")]
pub mod battery_degradation;
#[cfg(feature = "applied")]
pub mod cannibalism;
#[cfg(feature = "physics")]
pub mod chaos;
#[cfg(feature = "climate")]
pub mod climate;
#[cfg(feature = "applied")]
pub mod clinical_trials;
#[cfg(feature = "pure_math")]
pub mod elliptic_curves;
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
#[cfg(feature = "applied")]
pub mod paradox_lab;
#[cfg(feature = "physics")]
pub mod quantum;
#[cfg(feature = "physics")]
pub mod relativity;
#[cfg(feature = "physics")]
pub mod solid_state;
pub mod traceability;

include!(concat!(env!("OUT_DIR"), "/generated_tabs.rs"));

/// A trait for defining a tab in the Math Explorer application.
///
/// This trait serves as the primary extension point for the GUI. Each mathematical domain
/// (e.g., MRI Physics, Game Theory) should implement this trait as a standalone struct.
///
/// # Static Module Declaration & Constructor Generation
///
/// Local tab modules under `src/tabs/` are declared statically with `pub mod` items (and matching
/// `#[cfg(feature = "...")]` gates) in `src/tabs/mod.rs` so `rustc` and `cargo fmt` traverse the
/// full tab module tree directly.
///
/// Do not manually edit `app/mod.rs` to wire tab constructors. At compile time, `build.rs`
/// verifies that `src/tabs/mod.rs` declares every local tab module, scans `src/tabs/` (and
/// workspace path dependencies) for `impl ExplorerTab for YourStruct` definitions, and generates
/// `instantiate_tabs()` and `instantiate_tab()`.
///
/// ## Configuration Attributes
///
/// You can configure how your tab is built and ordered by adding header comments at the top of your module file:
///
/// - `// @explorer_feature = "feature_name"`: Gates constructor registration behind the matching Cargo feature (pair with `#[cfg(feature = "feature_name")]` on `pub mod` in `tabs/mod.rs`).
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
            "Algorithmic Information"
            | "Analysis & Calculus"
            | "Financial Math"
            | "Geometry & Topology"
            | "Graph Theory"
            | "Elliptic Curves"
            | "Number Theory" => "Pure Math",

            "Astrophysics"
            | "Chaos Theory"
            | "Fluid Dynamics"
            | "General Relativity"
            | "Medical Physics"
            | "MRI Physics"
            | "Quantum Mechanics"
            | "Solid State Physics" => "Physics",

            "Battery Degradation"
            | "Clinical Trials"
            | "Favoritism"
            | "Game Theory"
            | "Paradox Lab"
            | "Population Structure"
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
            "Algorithmic Information" => "algorithmic_information",
            "Analysis & Calculus" => "analysis",
            "Astrophysics" => "astrophysics",
            "Battery Degradation" => "battery_degradation",
            "Population Structure" => "cannibalism",
            "Chaos Theory" => "chaos",
            "Climate" => "climate",
            "Clinical Trials" => "clinical_trials",
            "Elliptic Curves" => "elliptic_curves",
            "Epidemiology" => "epidemiology",
            "Favoritism" => "favoritism",
            "Financial Math" => "financial_math",
            "Fluid Dynamics" => "fluid_dynamics",
            "Game Theory" => "game_theory",
            "General Relativity" => "relativity",
            "Geometry & Topology" => "geometry_topology",
            "Graph Theory" => "graph_theory",
            "Medical Physics" => "medical",
            "Morphogenesis" => "morphogenesis",
            "MRI Physics" => "mri",
            "Neuroscience" => "neuroscience",
            "Number Theory" => "number_theory",
            "Paradox Lab" => "paradox_lab",
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

    /// Encodes a `#tab=<slug>...` URL fragment for this tab (including active story or challenge IDs when present).
    fn encode_url_share_fragment(&self) -> Result<String, String> {
        if let Some(fw) = self.framework() {
            return fw.encode_share_fragment_for_tab(self.url_slug());
        }
        let preset_json = self.export_url_preset_json()?;
        crate::url_state::encode_share_fragment(
            self.url_slug(),
            self.active_tool_id(),
            preset_json.as_deref(),
        )
    }

    /// Validates and applies `ShareUrlParams` (supporting `&preset=`, `&challenge=`, `&step=`, and `&state=`).
    fn import_url_share_params(
        &mut self,
        params: &crate::url_state::ShareUrlParams,
        ctx: Option<&egui::Context>,
    ) -> Result<(), String> {
        if params.preset.is_none() && params.challenge.is_none() {
            return self
                .import_url_preset_json(params.tool.as_deref(), params.state_json.as_deref());
        }
        if let Some(fw) = self.framework_mut() {
            return fw.import_url_share_params(params, ctx);
        }
        Err(format!(
            "Tab '{}' does not support story presets or challenges",
            self.name()
        ))
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
        assert!(
            generated_tabs.contains("pub fn instantiate_tab("),
            "generated_tabs.rs should define instantiate_tab"
        );
        assert!(
            !generated_tabs.contains("include!("),
            "generated_tabs.rs must not emit inline include! module wrappers"
        );
        assert!(
            !generated_tabs.contains("pub mod "),
            "generated_tabs.rs must not emit inline pub mod wrappers"
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
            "src/tabs/mod.rs",
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

    fn collect_static_mod_gates(
        mod_rs: &str,
    ) -> std::collections::BTreeMap<String, Option<String>> {
        let mut gates = std::collections::BTreeMap::new();
        let mut active_cfg = None;
        for line in mod_rs.lines() {
            let clean = line.split("//").next().unwrap_or("").trim();
            if clean.is_empty() {
                continue;
            }
            if let Some(feat) = clean
                .strip_prefix("#[cfg(feature = \"")
                .and_then(|s| s.strip_suffix("\")]"))
            {
                active_cfg = Some(feat.to_string());
            } else if let Some(name) = clean
                .strip_prefix("pub mod ")
                .and_then(|s| s.strip_suffix(';'))
            {
                gates.insert(name.trim().to_string(), active_cfg.take());
            } else if !clean.starts_with("#[") {
                active_cfg = None;
            }
        }
        gates
    }

    #[test]
    fn test_static_tab_modules_match_filesystem_tabs_and_feature_gates() {
        let tabs_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tabs");
        let mod_rs = fs::read_to_string(tabs_dir.join("mod.rs")).expect("read src/tabs/mod.rs");
        let declared = collect_static_mod_gates(&mod_rs);
        let mut discovered = std::collections::BTreeMap::new();

        for entry in fs::read_dir(&tabs_dir).expect("read src/tabs").flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if matches!(
                name.as_str(),
                "mod.rs" | "generated_tabs.rs" | "generated.rs"
            ) {
                continue;
            }
            let (mod_name, target) = if path.is_dir() {
                (name, path.join("mod.rs"))
            } else if let Some(stem) = name.strip_suffix(".rs") {
                (stem.to_string(), path)
            } else {
                continue;
            };
            let Ok(src) = fs::read_to_string(&target) else {
                continue;
            };
            let has_tab_impl = src.lines().any(|l| {
                l.contains("impl ExplorerTab for ")
                    || l.contains("impl crate::tabs::ExplorerTab for ")
            });
            if !has_tab_impl {
                continue;
            }
            let feature = src.lines().find_map(|l| {
                let (_, rest) = l.split_once("// @explorer_feature = \"")?;
                let (feat, _) = rest.split_once('"')?;
                Some(feat.to_string())
            });
            discovered.insert(mod_name, feature);
        }

        assert!(
            discovered.len() >= 23,
            "Expected at least 23 local tabs, found {}",
            discovered.len()
        );
        assert_eq!(
            declared, discovered,
            "Static `pub mod` declarations in src/tabs/mod.rs must match local ExplorerTab files and @explorer_feature gates"
        );
    }
}
