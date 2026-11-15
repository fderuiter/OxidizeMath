# 17. MathBox Formula Rendering and Theory Metadata Equation Extension

Date: 2026-11-14

## Status

Accepted

## Context

Educational simulation tabs across `math_explorer_gui` previously displayed theoretical governing equations as unformatted plain text. The `TheoryDescribable` trait contract in `crates/scientific_metadata/src/theory.rs` provided descriptions, citations, and parameter constraints, but lacked a dedicated method to expose mathematical equations as LaTeX strings. Consequently, the theory portal and simulation control sidebars relied on plain text labels or 1D layout jobs instead of utilizing the 2D `MathBox` layout engine in `egui_plot::latex`. In addition, external citations referenced in `papers/partitions_implementation.tex` were missing from `papers/references.bib`.

## Decision

1. **Scientific Metadata Extension**: Extended `TheoryDescribable` in `crates/scientific_metadata/src/theory.rs` with `fn theory_latex_equation(&self) -> Option<String> { None }`. The default implementation returning `None` maintains full backwards compatibility with all existing mathematical models and visualizer components.
2. **2D MathBox Formula Rendering**: Integrated `render_math_label` from `egui_plot::latex` into `render_theory_summary_with_export` and `render_formula_with_export` within `math_explorer_gui/src/reflective_ui/mod.rs`. When a LaTeX string is present, the formula is laid out and rendered using the 2D `MathBox` engine, falling back gracefully to plain text labels if syntax is unsupported.
3. **Theory Portal Integration**: Updated `show_theory_portal` in `math_explorer_gui/src/framework/ui.rs` to extract `theory.theory_latex_equation()` and supply it directly to the export and rendering pipeline.
4. **ODE Solver Presets & Citations**: Implemented `theory_latex_equation`, enhanced `theory_description`, and accurate literature citations (`theory_citation`) across all presets (`Exponential`, `HarmonicOscillator`, `LogisticGrowth`) in `OdeSolverTool` in `math_explorer_gui/src/tabs/analysis/ode.rs`.
5. **Academic Bibliography Resolution**: Added verified BibTeX entries for `pushpa2022eisenstein` and `nath2025arithmetic` to `papers/references.bib`.

## Consequences

- **Positive:** Rich 2D mathematical formula rendering in theory portals and simulation parameter sidebars without breaking compatibility.
- **Positive:** Preserved fast slider response without UI thread lag during interactive manipulation.
- **Positive:** Resolved partition paper citation dependencies with complete bibliographic metadata.
