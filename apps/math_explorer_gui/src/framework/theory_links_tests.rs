#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::theory_links::{resolve_citation_reference_links, resolve_portal_paper_links};
use super::*;
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::StateSnapshot;
use crate::framework::recovery::FiniteStateGuard;
use eframe::egui;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::{BTreeSet, HashMap};

struct DummyUnifiedEquationModel {
    val: f64,
}

impl TheoryDescribable for DummyUnifiedEquationModel {
    fn theory_description(&self) -> String {
        r"Diffusion rate $\frac{\partial u}{\partial t}$ in $\mathbb{R}^2$.".to_string()
    }
    fn theory_latex_equation(&self) -> Option<String> {
        Some(r"\frac{\partial u}{\partial t} = D \nabla^2 u".to_string())
    }
    fn phonetic_description(&self) -> String {
        "d u by d t equals D del squared u".to_string()
    }
    fn theory_citation(&self) -> String {
        "Attention Is All You Need (Vaswani et al., 2017)".to_string()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

impl FiniteStateGuard for DummyUnifiedEquationModel {
    fn is_finite_state(&self) -> bool {
        self.val.is_finite()
    }
}

impl UnifiedModel for DummyUnifiedEquationModel {
    fn new(params: &HashMap<String, f64>) -> Self {
        Self {
            val: params.get("d").copied().unwrap_or(1.0),
        }
    }
    fn step(&mut self, _params: &HashMap<String, f64>) {}
    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 1,
            height: 1,
            pixels: std::sync::Arc::new(std::sync::RwLock::new(vec![eframe::egui::Color32::BLACK])),
            custom_data: Vec::new(),
            structured_data: None,
            sonification: crate::audio::SonificationFrame::default(),
        }
    }
    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        map.insert(
            "d".to_string(),
            ParameterConstraint {
                min: 0.1,
                max: 2.0,
                step: 0.1,
            },
        );
        map
    }
    fn name() -> &'static str {
        "DummyUnifiedEquationModel"
    }
    fn domain() -> &'static str {
        "analysis"
    }
}

#[test]
fn test_unified_sim_tool_forwards_theory_latex_equation() {
    let tool = UnifiedSimTool::<DummyUnifiedEquationModel>::new();
    assert_eq!(
        tool.theory().theory_latex_equation().as_deref(),
        Some(r"\frac{\partial u}{\partial t} = D \nabla^2 u")
    );
}

#[test]
fn test_resolve_portal_paper_and_bibtex_links_filters_specs() {
    let spin_links = resolve_portal_paper_links("SpinVisualizer", "quantum");
    assert!(!spin_links.is_empty());
    assert_eq!(spin_links[0].paper_name, "quantum_mechanics.tex");
    assert!(spin_links[0]
        .pdf_url
        .as_deref()
        .is_some_and(|u| u.ends_with("/quantum_mechanics.pdf")));
    assert!(spin_links[0]
        .tex_url
        .as_deref()
        .is_some_and(|u| u.ends_with("/papers/quantum_mechanics.tex")));

    let ai_links = resolve_portal_paper_links("TrainingMonitorTool", "ai");
    assert!(
        ai_links.is_empty(),
        "TrainingMonitorTool (ai) must filter spec: targets without fallbacks: {ai_links:?}"
    );

    let fractran_links = resolve_portal_paper_links("FractranMachineTool", "number_theory");
    assert_eq!(
        fractran_links[0].paper_name,
        "algorithmic_information_rust.tex"
    );

    let glicko_links = resolve_portal_paper_links("Glicko2TournamentTool", "game_theory");
    assert!(
        glicko_links.is_empty(),
        "Unexpected fallback: {glicko_links:?}"
    );
    let net_metrics_links = resolve_portal_paper_links("NetworkMetricsTool", "graph_theory");
    assert_eq!(net_metrics_links[0].paper_name, "graph_parameters_rust.tex");

    let astro_links = resolve_portal_paper_links("GalacticOrbitSimulator", "astrophysics");
    assert_eq!(
        astro_links[0].paper_name,
        "dwarf_galaxy_empirical_dependencies.tex"
    );

    let volumetric_links = resolve_portal_paper_links("VolumetricFieldStudioTool", "quantum");
    assert!(volumetric_links
        .iter()
        .any(|l| l.paper_name == "gaussian_splatting.tex"));
    assert!(volumetric_links
        .iter()
        .any(|l| l.paper_name == "isosurface_extraction.tex"));
    assert!(volumetric_links
        .iter()
        .any(|l| l.paper_name == "quantum_mechanics.tex"));

    let cera_ext = resolve_citation_reference_links(
        "CERA: A Framework for Improved Generalization of Machine Learning Models to Changed Climates (Liu et al., 2025)",
    );
    assert!(cera_ext
        .iter()
        .any(|e| e.url == "https://arxiv.org/abs/2509.00010"));

    let attn_ext =
        resolve_citation_reference_links("Attention Is All You Need (Vaswani et al., 2017)");
    assert!(attn_ext
        .iter()
        .any(|e| e.url == "https://arxiv.org/abs/1706.03762"));

    let part_ext = resolve_citation_reference_links(
        "On Eisenstein series, color partition and divisor function (Pushpa & Vasuki, 2022)",
    );
    assert!(part_ext
        .iter()
        .any(|e| e.url == "https://doi.org/10.1007/s40065-022-00360-6"));
}

#[test]
fn test_win_ratio_workbench_portal_paper_links() {
    for tool_id in [
        "WinRatioWorkbenchTool",
        "WinRatioStressTesterTool",
        "WinRatioTool",
    ] {
        let win_ratio_links = resolve_portal_paper_links(tool_id, "clinical_trials");
        assert!(!win_ratio_links.is_empty());
        assert_eq!(win_ratio_links[0].paper_name, "win_ratio.tex");
        assert!(win_ratio_links
            .iter()
            .any(|l| l.paper_name == "clinical_trials_statistics.tex"));
    }
}

#[test]
fn test_cortical_morphometry_resolves_freesurfer_paper_and_doi_links() {
    let cortical_links = resolve_portal_paper_links("CorticalMorphometryTool", "neuroscience");
    assert!(!cortical_links.is_empty());
    assert_eq!(cortical_links[0].paper_name, "freesurfer.tex");
    let ext = resolve_citation_reference_links(
        "Cortical Surface-Based Analysis I: Segmentation and Surface Reconstruction (Dale, Fischl & Sereno, 1999, https://doi.org/10.1006/nimg.1998.0395)",
    );
    assert!(ext
        .iter()
        .any(|e| e.url == "https://doi.org/10.1006/nimg.1998.0395"));
}

const FORBIDDEN_RAW_TEX: &[&str] = &[
    "\\",
    r"\hbar",
    r"\langle",
    r"\rangle",
    r"\vec",
    r"\hat",
    r"\widehat",
    r"\bar",
    r"\dot",
    r"\ddot",
    r"\tilde",
    r"\widetilde",
    r"\wedge",
    r"\top",
    r"\bot",
    r"\mapsto",
    r"\implies",
    r"\neg",
    r"\subseteq",
    r"\supseteq",
    r"\mathbb",
    r"\left",
    r"\right",
    r"\frac",
    r"\text",
    r"\mathrm",
    r"\mathbf",
    r"\AA",
    r"\deg",
    r"\dots",
    r"\pmod",
    "──",
    "ddot",
    "widehat",
    "widetilde",
];

fn assert_clean_portal_output(
    tool_id: &str,
    domain: &str,
    out: &egui::FullOutput,
) -> (usize, usize) {
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must exist");
    let (mut paper_links, mut ext_links) = (0_usize, 0_usize);
    for (_, node) in &ak.nodes {
        for text in node.label().into_iter().chain(node.value()) {
            for raw in FORBIDDEN_RAW_TEX {
                assert!(
                    !text.contains(raw),
                    "Tool `{tool_id}` rendered corrupted TeX/bar token `{raw}` in `{text}`"
                );
            }
            assert!(
                !text.contains("spec:") && !text.contains("registry:"),
                "Tool `{tool_id}` leaked spec:/registry: token in `{text}`"
            );
            if domain != "favoritism" {
                assert!(
                    !text.contains("favorite_child.pdf") && !text.contains("favorite_child.tex"),
                    "Tool `{tool_id}` ({domain}) leaked favorite_child link in `{text}`"
                );
            }
        }
        if let Some(lbl) = node.label() {
            if lbl.starts_with("📄 View PDF") || lbl.starts_with("📝 View .tex Source") {
                paper_links += 1;
            } else if lbl.starts_with("🔗 ") {
                ext_links += 1;
            }
        }
    }
    assert!(
        paper_links == 0 || (paper_links >= 2 && paper_links % 2 == 0),
        "Tool `{tool_id}` must expose paired PDF + .tex hyperlinks when matched, got {paper_links}"
    );
    assert!(
        paper_links + ext_links >= 1 && (domain == "favoritism" || ext_links >= 1),
        "Tool `{tool_id}` ({domain}) must expose portal hyperlinks (paper={paper_links}, ext={ext_links})"
    );
    (paper_links, ext_links)
}

fn find_hyperlink_center(out: &egui::FullOutput, prefix: &str) -> Option<egui::Pos2> {
    let ak = out.platform_output.accesskit_update.as_ref()?;
    let bounds = ak.nodes.iter().find_map(|(_, n)| {
        n.label()
            .filter(|l| l.starts_with(prefix))
            .and_then(|_| n.bounds())
    })?;
    Some(egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    ))
}

fn run_portal_frame(
    ctx: &egui::Context,
    framework: &mut SimulationFramework,
    screen_rect: Option<egui::Rect>,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    ctx.run(
        egui::RawInput {
            screen_rect,
            events,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                framework.show_theory_portal(ui, "portal_verify");
            });
        },
    )
}

fn assert_portal_hyperlink_opens_url(
    framework: &mut SimulationFramework,
    screen_rect: Option<egui::Rect>,
    prefix: &str,
    expected_url_substr: &str,
) {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let initial = run_portal_frame(&ctx, framework, screen_rect, Vec::new());
    let pos = find_hyperlink_center(&initial, prefix)
        .unwrap_or_else(|| panic!("Expected portal hyperlink starting with `{prefix}`"));
    let press_events = vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = run_portal_frame(&ctx, framework, screen_rect, press_events);
    let release_events = vec![egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    }];
    let release = run_portal_frame(&ctx, framework, screen_rect, release_events);
    let opened = release
        .platform_output
        .commands
        .iter()
        .any(|cmd| match cmd {
            egui::OutputCommand::OpenUrl(open) => open.url.contains(expected_url_substr),
            _ => false,
        });
    assert!(
        opened,
        "Clicking `{prefix}` must emit OpenUrl containing `{expected_url_substr}`, got {:?}",
        release.platform_output.commands
    );
}

#[test]
fn test_all_registered_tools_render_clean_portal_latex_and_hyperlinks() {
    let domains: BTreeSet<&'static str> = inventory::iter::<ToolMetadata>
        .into_iter()
        .map(|m| m.domain)
        .collect();
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1440.0, 900.0),
    ));
    let mut verified_tools = 0_usize;

    for domain in domains {
        let mut framework = SimulationFramework::new(domain);
        framework.show_theory_portal = true;
        for idx in 0..framework.available_tools.len() {
            framework.select_tool(idx);
            framework.show_theory_portal = true;
            let tool_id = framework.available_tools[idx].id;
            let ctx = egui::Context::default();
            ctx.enable_accesskit();
            let out = run_portal_frame(&ctx, &mut framework, screen_rect, Vec::new());
            let _ = assert_clean_portal_output(tool_id, domain, &out);
            if tool_id == "SpinVisualizer" {
                assert_portal_hyperlink_opens_url(
                    &mut framework,
                    screen_rect,
                    "📄 View PDF",
                    "quantum_mechanics.pdf",
                );
            } else if tool_id == "CeraTool" {
                assert_portal_hyperlink_opens_url(
                    &mut framework,
                    screen_rect,
                    "🔗 arXiv:2509.00010",
                    "https://arxiv.org/abs/2509.00010",
                );
            }
            verified_tools += 1;
        }
    }

    assert_verified_tool_count(verified_tools);
}

fn assert_verified_tool_count(verified_tools: usize) {
    if cfg!(all(
        feature = "ai",
        feature = "physics",
        feature = "pure_math"
    )) {
        assert!(
            verified_tools >= 65,
            "Expected >= 65 tools, got {verified_tools}"
        );
    } else if cfg!(any(
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "climate",
        feature = "engineering",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math",
    )) {
        assert!(verified_tools > 0);
    } else {
        assert_eq!(verified_tools, 0);
    }
}

#[test]
fn test_bibtex_key_matching_disambiguation_and_zero_favorite_child_leaks() {
    for (tool_id, domain) in [
        ("MriTool", "mri"),
        ("MriKSpaceScannerTool", "mri"),
        ("BeamProfilingTool", "medical"),
        ("DoseCalculationTool", "medical"),
        ("Co2ProjectionsTool", "climate"),
        ("HyperbolicDiskTool", "hyperbolic_geometry"),
        ("MarkovSequenceWorkbenchTool", "game_theory"),
        ("CapacityFadeTool", "battery_degradation"),
    ] {
        let links = resolve_portal_paper_links(tool_id, domain);
        assert!(
            links.is_empty(),
            "{tool_id} ({domain}) must not inherit unrelated fallback PDFs, got {links:?}"
        );
    }
    let fav_links = resolve_portal_paper_links("FavoritismTool", "favoritism");
    assert_eq!(fav_links.len(), 1);
    assert_eq!(fav_links[0].paper_name, "favorite_child.tex");

    let optics_ext = resolve_citation_reference_links(
        "Born & Wolf (1999), Principles of Optics (born1999principles); Luneburg (1964) (luneburg1964mathematical); Berry & Upstill (1980) (berry1980catastrophe)",
    );
    assert!(optics_ext
        .iter()
        .any(|e| e.url.contains("10.1017/CBO9781139644181")));
    assert!(optics_ext
        .iter()
        .any(|e| e.url.contains("10.1525/9780520328266")));
    assert!(optics_ext
        .iter()
        .any(|e| e.url.contains("10.1016/S0079-6638(08)70215-4")));

    let bh_ext = resolve_citation_reference_links(
        "Schwarzschild (1916); Misner, Thorne & Wheeler (1973), Gravitation; Chandrasekhar (1983), The Mathematical Theory of Black Holes",
    );
    assert!(
        bh_ext.len() >= 3,
        "Expected >= 3 black hole citation links, got {bh_ext:?}"
    );

    let ambig_bare = resolve_citation_reference_links("Lutz et al. (2024)");
    assert!(
        ambig_bare.is_empty(),
        "Ambiguous surname+year without co-author or title word must not guess: {ambig_bare:?}"
    );
    let lutz_qi = resolve_citation_reference_links("Lutz, Qi & Yu (2024)");
    assert_eq!(lutz_qi.len(), 1);
    let lutz_stull = resolve_citation_reference_links("Lutz & Stull (2024)");
    assert_eq!(lutz_stull.len(), 1);
    assert_ne!(lutz_qi[0].url, lutz_stull[0].url);
}
