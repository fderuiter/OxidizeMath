#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! `TheoryDescribable`, `ToolMetadata`, and `StarterPresetMetadata` registrations for the
//! `Optimization` tab workbenches.

use super::constrained_tool::ConstrainedContourTool;
use super::lp_tool::LinearProgrammingTool;
use super::pareto_tool::ParetoFrontierTool;
use super::qp_tool::QuadraticProgrammingTool;
use std::collections::HashMap;

impl scientific_metadata::theory::TheoryDescribable for LinearProgrammingTool {
    fn theory_description(&self) -> String {
        "Two-phase Simplex method for $\\max c^T x$ subject to $Ax \\le b, x \\ge 0$, using Bland's smallest-index pivot rule to guarantee finite termination without cycling and extracting optimal dual shadow prices $y^* = B^{-T} c_B$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Two-phase Simplex linear programming with Bland's anti-cycling rule and shadow prices"
            .into()
    }
    fn theory_citation(&self) -> String {
        "Dantzig (1963) Linear Programming and Extensions; Bland (1977) New Finite Pivoting Rules"
            .into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            ("c".into(), "Linear objective gradient vector c".into()),
            (
                "shadow_prices".into(),
                "Dual multipliers y* measuring marginal RHS sensitivity".into(),
            ),
        ])
    }
}

impl scientific_metadata::theory::TheoryDescribable for QuadraticProgrammingTool {
    fn theory_description(&self) -> String {
        "Primal active-set method for strictly convex quadratic programs $\\min \\frac{1}{2} x^T Q x + c^T x$ subject to $A_{eq} x = b_{eq}$ and $A_{ineq} x \\le b_{ineq}$, solving equality-constrained KKT subproblems and dropping negative inequality multipliers $\\mu_i < 0$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Active-set convex quadratic programming with Karush-Kuhn-Tucker multipliers".into()
    }
    fn theory_citation(&self) -> String {
        "Nocedal & Wright (2006) Numerical Optimization, Chapter 16 (Quadratic Programming)".into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "Q".into(),
                "Symmetric positive-definite objective Hessian matrix".into(),
            ),
            (
                "mu".into(),
                "Non-negative KKT dual multipliers on active linear inequalities".into(),
            ),
        ])
    }
}

impl scientific_metadata::theory::TheoryDescribable for ConstrainedContourTool {
    fn theory_description(&self) -> String {
        "Nonlinear constrained optimization comparing Powell-Hestenes-Rockafellar shifted Augmented Lagrangian $\\mathcal{L}_\\rho(x,\\lambda,\\mu)$, limited-memory quasi-Newton `L-BFGS-B` with Cauchy active-bound identification, spectral Projected Gradient, and derivative-free Nelder-Mead simplex search.".into()
    }
    fn phonetic_description(&self) -> String {
        "Nonlinear constrained contour optimization with L-BFGS-B and Augmented Lagrangian".into()
    }
    fn theory_citation(&self) -> String {
        "Byrd, Lu, Nocedal & Zhu (1995) L-BFGS-B; Birgin & Martínez (2014) Practical Augmented Lagrangian Methods; Huang et al. (2023) LoraHub".into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "box_bounds".into(),
                "Coordinate-wise lower and upper hyperrectangle bounds".into(),
            ),
            (
                "circle_radius".into(),
                "Nonlinear disk or ring constraint radius r".into(),
            ),
        ])
    }
}

impl scientific_metadata::theory::TheoryDescribable for ParetoFrontierTool {
    fn theory_description(&self) -> String {
        "Multi-objective Pareto optimization comparing convex weighted-sum scalarization $\\min \\sum w_i f_i(x)$ against $\\epsilon$-constraint scalarization $\\min f_1(x)$ s.t. $f_2(x) \\le \\epsilon$ (which recovers non-convex Pareto dents) alongside FISTA proximal LASSO trade-off curves.".into()
    }
    fn phonetic_description(&self) -> String {
        "Multi-objective Pareto frontier extraction and FISTA sparse regularization trade-off"
            .into()
    }
    fn theory_citation(&self) -> String {
        "Miettinen (1998) Nonlinear Multiobjective Optimization; Beck & Teboulle (2009) FISTA"
            .into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "concavity_amp".into(),
                "Amplitude of non-convex Pareto frontier dent".into(),
            ),
            (
                "num_samples".into(),
                "Number of scalarized sweep subproblems".into(),
            ),
        ])
    }
}

const LP_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Resource Allocation Polytope","domain":"optimization","timestamp":"2026-11-23T00:00:00Z"},"config":{"c":[4.0,3.0],"maximize":true,"constraints":[[2.0,1.0,20.0],[1.0,2.0,22.0],[3.0,1.0,27.0]],"senses":[0,0,0]}}"#;
const QP_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Equality-Constrained Convex QP","domain":"optimization","timestamp":"2026-11-23T00:00:00Z"},"config":{"target":[3.8,3.2],"q_diag":[2.4,1.6],"q_offdiag":0.4,"ineq":[[1.0,1.0,4.5],[1.0,-0.4,2.4]],"eq_enabled":true,"eq_line":[1.0,1.5,5.2]}}"#;
const CONSTRAINED_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Rosenbrock Disk Barrier (Augmented Lagrangian)","domain":"optimization","timestamp":"2026-11-23T00:00:00Z"},"config":{"start":[-1.2,1.0],"box_bounds":[-1.5,0.9,-0.8,1.5],"circle_radius":0.95,"circle_equality":false,"rosenbrock":true,"solver_idx":0}}"#;
const PARETO_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Non-Convex Pareto Dent (ε-Constraint)","domain":"optimization","timestamp":"2026-11-23T00:00:00Z"},"config":{"concavity_amp":0.48,"problem_mode":0,"num_samples":25,"selected_idx":11}}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LinearProgrammingTool",
        display_name: "Linear Programming (Simplex)",
        domain: "optimization",
        tags: &["simplex", "linear-programming", "duality", "shadow-prices"],
        build: || Box::new(LinearProgrammingTool::default()),
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "QuadraticProgrammingTool",
        display_name: "Quadratic Programming (Active-Set)",
        domain: "optimization",
        tags: &["qp", "active-set", "kkt", "convex"],
        build: || Box::new(QuadraticProgrammingTool::default()),
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ConstrainedContourTool",
        display_name: "Nonlinear Constrained Contours",
        domain: "optimization",
        tags: &["lbfgsb", "augmented-lagrangian", "projected-gradient", "nelder-mead"],
        build: || Box::new(ConstrainedContourTool::default()),
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ParetoFrontierTool",
        display_name: "Pareto Multi-Objective Workbench",
        domain: "optimization",
        tags: &["pareto", "multi-objective", "fista", "lasso"],
        build: || Box::new(ParetoFrontierTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "opt_lp_resource_polytope",
        title: "Resource Allocation Polytope",
        description: "Two-phase Simplex vertex pivot walk with Bland's rule and dual shadow prices.",
        category: "Linear & Convex",
        icon: "📐",
        domain: "optimization",
        tool_name: Some("LinearProgrammingTool"),
        preset_generator: || LP_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "opt_qp_active_set_kkt",
        title: "Equality-Constrained Convex QP",
        description: "Active-set quadratic bowl projection with active KKT Lagrange multiplier vectors.",
        category: "Linear & Convex",
        icon: "🎯",
        domain: "optimization",
        tool_name: Some("QuadraticProgrammingTool"),
        preset_generator: || QP_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "opt_constrained_rosenbrock_disk",
        title: "Rosenbrock Disk Barrier",
        description: "Shifted Augmented Lagrangian & L-BFGS-B navigating Rosenbrock under disk + box constraints.",
        category: "Nonlinear Optimization",
        icon: "🧭",
        domain: "optimization",
        tool_name: Some("ConstrainedContourTool"),
        preset_generator: || CONSTRAINED_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "opt_pareto_nonconvex_dent",
        title: "Non-Convex Pareto Dent",
        description: "ε-Constraint scalarization recovering non-convex Pareto solutions missed by weighted sums.",
        category: "Multi-Objective",
        icon: "⚖️",
        domain: "optimization",
        tool_name: Some("ParetoFrontierTool"),
        preset_generator: || PARETO_PRESET_JSON,
    }
}

// [cite:lorahub]
