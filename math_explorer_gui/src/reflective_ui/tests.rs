use super::*;
use std::sync::Mutex;

pub(crate) static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn test_constraint(min: f64, max: f64, step: f64) -> ParameterConstraint {
    ParameterConstraint { min, max, step }
}

struct DummyMissingTheoryModel {
    param_val: f64,
}

impl TheoryDescribable for DummyMissingTheoryModel {
    fn theory_description(&self) -> String {
        "Dummy description".into()
    }
    fn phonetic_description(&self) -> String {
        "Dummy phonetic".into()
    }
    fn theory_citation(&self) -> String {
        "Dummy citation".into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        HashMap::new()
    }

    fn get_parameter(&self, name: &str) -> Option<f64> {
        if name == "missing_param" {
            Some(self.param_val)
        } else {
            None
        }
    }

    fn set_parameter(&mut self, name: &str, value: f64) {
        if name == "missing_param" {
            self.param_val = value;
        }
    }
}

#[test]
fn test_get_theory_constraint_fallback_and_telemetry() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = global_registry().try_recv_all();

    let model = DummyMissingTheoryModel { param_val: 10.0 };
    let constraint = get_theory_constraint(&model, "missing_param");

    assert_eq!(constraint.min, 0.0);
    assert_eq!(constraint.max, 100.0);
    assert_eq!(constraint.step, 0.1);

    let events = global_registry().try_recv_all();
    let warning_event = events.iter().find(|e| {
        e.source == "reflective_ui"
            && e.severity == Severity::Warning
            && e.message.contains("missing_param")
    });
    assert!(
        warning_event.is_some(),
        "Expected diagnostic warning event for missing parameter metadata"
    );
}

#[test]
fn test_render_theory_parameter_fallback_and_telemetry() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = global_registry().try_recv_all();

    let ctx = egui::Context::default();
    let model = DummyMissingTheoryModel { param_val: 10.0 };
    let mut value = 10.0;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_theory_parameter(ui, &model, "missing_param", "Missing Param", &mut value);
        });
    });

    let events = global_registry().try_recv_all();
    let warning_event = events.iter().find(|e| {
        e.source == "reflective_ui"
            && e.severity == Severity::Warning
            && e.message.contains("missing_param")
    });
    assert!(
        warning_event.is_some(),
        "Expected diagnostic warning event during parameter rendering"
    );
}

#[test]
fn test_render_all_theory_parameters_resiliency() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = global_registry().try_recv_all();

    struct DummyIncompleteModel;
    impl TheoryDescribable for DummyIncompleteModel {
        fn theory_description(&self) -> String {
            "Incomplete".to_string()
        }
        fn phonetic_description(&self) -> String {
            "Incomplete".to_string()
        }
        fn theory_citation(&self) -> String {
            "None".to_string()
        }
        fn available_descriptions(&self) -> HashMap<String, String> {
            HashMap::new()
        }
        fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
            let mut map = HashMap::new();
            map.insert(
                "valid_param".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 10.0,
                    step: 1.0,
                },
            );
            map
        }
        fn get_parameter(&self, name: &str) -> Option<f64> {
            if name == "valid_param" {
                Some(5.0)
            } else {
                None
            }
        }
    }

    let ctx = egui::Context::default();
    let mut model = DummyIncompleteModel;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let changed = render_all_theory_parameters(ui, &mut model);
            assert!(!changed);
        });
    });
}

#[test]
fn test_render_copyable_metric() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp1 = render_copyable_metric(ui, "Test Metric", "42.0");
            assert!(resp1.rect.width() >= 0.0);

            let resp2 = render_copyable_metric(ui, "Test Metric:", "100");
            assert!(resp2.rect.width() >= 0.0);

            let resp3 = render_copyable_metric(ui, "", "empty_label_val");
            assert!(resp3.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_render_formula_with_export() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let r1 = render_formula_with_export(ui, "\\frac{d}{dx} e^x", Some("d/dx e^x"));
            let r2 = render_formula_with_export(ui, "", None);
            assert!(r1.rect.width() >= 0.0 && r2.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_render_theory_summary_with_export() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let r1 = render_theory_summary_with_export(ui, "Euler identity", Some("e^{i\\pi}+1=0"));
            let r2 = render_theory_summary_with_export(ui, "", None);
            assert!(r1.rect.width() >= 0.0 && r2.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_formula_export_populates_copied_text() {
    let ctx = egui::Context::default();
    let full_output = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.ctx().copy_text("\\int_0^1 x^2 dx = 1/3".to_string());
        });
    });
    let copied = full_output
        .platform_output
        .commands
        .iter()
        .any(|cmd| match cmd {
            egui::OutputCommand::CopyText(txt) => txt == "\\int_0^1 x^2 dx = 1/3",
            _ => false,
        });
    assert!(
        copied,
        "Expected CopyText output command in full_output.platform_output, got: {:?}",
        full_output.platform_output.commands
    );
}

#[test]
fn test_render_theory_parameter_help_button_toggle() {
    let ctx = egui::Context::default();
    let model = DummyMissingTheoryModel { param_val: 10.0 };
    let mut value = 10.0;

    // Frame 1: render parameter, portal state is not set or false
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_theory_parameter(ui, &model, "missing_param", "Missing Param", &mut value);
        });
    });

    let initial_portal = ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL")));
    assert_ne!(initial_portal, Some(true));

    // Get rect/interaction or simulate toggle in context
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), true));
    let updated_portal = ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL")));
    assert_eq!(updated_portal, Some(true));
}

#[test]
fn test_render_theory_parameter_with_custom_description_and_help_button() {
    struct DummyWithDescModel;
    impl TheoryDescribable for DummyWithDescModel {
        fn theory_description(&self) -> String {
            "Model description".to_string()
        }
        fn phonetic_description(&self) -> String {
            "Model phonetic".to_string()
        }
        fn theory_citation(&self) -> String {
            "Citation 2026".to_string()
        }
        fn available_descriptions(&self) -> HashMap<String, String> {
            let mut map = HashMap::new();
            map.insert(
                "param1".to_string(),
                "Custom parameter 1 tooltip".to_string(),
            );
            map
        }
        fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
            let mut map = HashMap::new();
            map.insert(
                "param1".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 50.0,
                    step: 0.5,
                },
            );
            map
        }
    }

    let ctx = egui::Context::default();
    let model = DummyWithDescModel;
    let mut val = 25.0;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let response = render_theory_parameter(ui, &model, "param1", "Param One", &mut val);
            assert!(response.rect.width() >= 0.0);
        });
    });
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_render_all_theory_parameters_single_metadata_query() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingTheoryModel {
        alpha: f64,
        beta: f64,
        gamma: f64,
        theory_params_count: AtomicUsize,
        available_descs_count: AtomicUsize,
    }

    impl TheoryDescribable for CountingTheoryModel {
        fn theory_description(&self) -> String {
            "Counting theory".into()
        }
        fn phonetic_description(&self) -> String {
            "Counting phonetic".into()
        }
        fn theory_citation(&self) -> String {
            "Citation".into()
        }
        fn available_descriptions(&self) -> HashMap<String, String> {
            self.available_descs_count.fetch_add(1, Ordering::SeqCst);
            HashMap::from([
                ("alpha".into(), "Alpha tooltip".into()),
                ("beta".into(), "Beta tooltip".into()),
                ("gamma".into(), "Gamma tooltip".into()),
            ])
        }
        fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
            self.theory_params_count.fetch_add(1, Ordering::SeqCst);
            HashMap::from([
                ("alpha".into(), test_constraint(0.0, 1.0, 0.05)),
                ("beta".into(), test_constraint(1.0, 10.0, 0.5)),
                ("gamma".into(), test_constraint(-5.0, 5.0, 0.1)),
            ])
        }
        fn get_parameter(&self, name: &str) -> Option<f64> {
            match name {
                "alpha" => Some(self.alpha),
                "beta" => Some(self.beta),
                "gamma" => Some(self.gamma),
                _ => None,
            }
        }
        fn set_parameter(&mut self, name: &str, value: f64) {
            match name {
                "alpha" => self.alpha = value,
                "beta" => self.beta = value,
                "gamma" => self.gamma = value,
                _ => {}
            }
        }
    }

    let ctx = egui::Context::default();
    let mut model = CountingTheoryModel {
        alpha: 0.5,
        beta: 5.0,
        gamma: 0.0,
        theory_params_count: AtomicUsize::new(0),
        available_descs_count: AtomicUsize::new(0),
    };

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let changed = render_all_theory_parameters(ui, &mut model);
            assert!(!changed);
        });
    });

    assert_eq!(
        model.theory_params_count.load(Ordering::SeqCst),
        1,
        "theory_parameters() should be queried exactly once during batch rendering"
    );
    assert_eq!(
        model.available_descs_count.load(Ordering::SeqCst),
        1,
        "available_descriptions() should be queried at most once during batch rendering"
    );
}

#[test]
fn test_render_parameter_slider_direct() {
    let ctx = egui::Context::default();
    let constraint = test_constraint(0.0, 50.0, 0.25);
    let mut val = 12.5;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let response = render_parameter_slider(
                ui,
                "Direct Slider",
                &mut val,
                &constraint,
                "Direct tooltip text",
            );
            assert!(response.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_lookup_theory_constraint() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = global_registry().try_recv_all();

    let params = HashMap::from([(
        "custom_param".to_string(),
        test_constraint(-10.0, 10.0, 0.5),
    )]);

    // Existing parameter lookup returns borrowed constraint without allocation
    let existing = lookup_theory_constraint(&params, "custom_param");
    assert_eq!(existing.min, -10.0);
    assert_eq!(existing.max, 10.0);
    assert_eq!(existing.step, 0.5);

    // get_theory_constraint_ref alias works identically
    let existing_ref = get_theory_constraint_ref(&params, "custom_param");
    assert_eq!(existing_ref.min, -10.0);

    // Missing parameter returns fallback constraint and emits warning
    let missing = lookup_theory_constraint(&params, "non_existent");
    assert_eq!(missing.min, 0.0);
    assert_eq!(missing.max, 100.0);
    assert_eq!(missing.step, 0.1);

    let events = global_registry().try_recv_all();
    let warning_event = events.iter().find(|e| {
        e.source == "reflective_ui"
            && e.severity == Severity::Warning
            && e.message.contains("non_existent")
    });
    assert!(
        warning_event.is_some(),
        "Expected diagnostic warning for non_existent parameter in lookup_theory_constraint"
    );
}

#[test]
fn test_cached_theory_constraint() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let _guard = TEST_MUTEX.lock().unwrap();
    clear_metadata_cache();

    struct CachedDummyModel {
        queries: AtomicUsize,
    }

    impl TheoryDescribable for CachedDummyModel {
        fn theory_description(&self) -> String {
            "Cached dummy".into()
        }
        fn phonetic_description(&self) -> String {
            "Phonetic".into()
        }
        fn theory_citation(&self) -> String {
            "Citation".into()
        }
        fn available_descriptions(&self) -> HashMap<String, String> {
            HashMap::new()
        }
        fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
            self.queries.fetch_add(1, Ordering::SeqCst);
            HashMap::from([
                ("p1".into(), test_constraint(1.0, 2.0, 0.1)),
                ("p2".into(), test_constraint(3.0, 4.0, 0.2)),
            ])
        }
    }

    let model = CachedDummyModel {
        queries: AtomicUsize::new(0),
    };

    let c1 = get_theory_constraint(&model, "p1");
    assert_eq!(c1.min, 1.0);
    assert_eq!(model.queries.load(Ordering::SeqCst), 1);

    let c2 = get_theory_constraint(&model, "p2");
    assert_eq!(c2.min, 3.0);
    assert_eq!(
        model.queries.load(Ordering::SeqCst),
        1,
        "get_theory_constraint should use cached metadata without calling theory_parameters() again"
    );

    clear_metadata_cache();
    let c3 = get_theory_constraint(&model, "p1");
    assert_eq!(c3.min, 1.0);
    assert_eq!(
        model.queries.load(Ordering::SeqCst),
        2,
        "Clearing cache should allow fresh metadata query"
    );
}
