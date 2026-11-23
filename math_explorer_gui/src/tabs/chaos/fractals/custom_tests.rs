#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::InteractiveTool;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

#[test]
fn test_custom_fractal_escape_zero_alloc() {
    let mut bar = default_fractal_input_bar();
    bar.set_expression("a * z^3 + c");
    let mut pmap = BTreeMap::new();
    pmap.insert("a".to_string(), 1.25);
    bar.set_params_map(&pmap);

    let job = CompiledFractalJob::from_bar(&bar);
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((it0, it1), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        (
            job.escape_time(Complex::new(0.0, 0.0), Complex::new(0.1, 0.1), 120),
            job.escape_time(Complex::new(0.0, 0.0), Complex::new(2.0, 1.5), 120),
        )
    });
    assert_eq!(it0, 120);
    assert!(it1 < 5);
}

#[test]
fn test_custom_fractal_progressive_worker() {
    let mut bar = default_fractal_input_bar();
    bar.set_expression("a * z^3 + c");
    let mut pmap = BTreeMap::new();
    pmap.insert("a".to_string(), 1.25);
    bar.set_params_map(&pmap);
    let job = Arc::new(CompiledFractalJob::from_bar(&bar));

    let req = FractalRenderRequest {
        generation: 1,
        width: 64,
        height: 48,
        center: Complex::new(0.0, 0.0),
        zoom: 1.0,
        max_iter: 80,
        mode: FractalMode::CustomMandelbrot,
        julia_c: Complex::new(-0.4, 0.6),
        high_contrast: false,
    };
    let shared_req = Arc::new(RwLock::new(FractalRenderRequest {
        generation: 0,
        width: 16,
        height: 12,
        ..req
    }));
    let shared_formula = Arc::new(RwLock::new(Some(Arc::clone(&job))));
    let latest_gen = Arc::new(AtomicU64::new(0));
    let mut runner = FractalRunner::with_formula(
        Arc::clone(&shared_req),
        Arc::clone(&latest_gen),
        Arc::clone(&shared_formula),
    );
    *shared_req.write().unwrap() = req;
    latest_gen.store(1, Ordering::Release);

    let mut strides = Vec::new();
    for _ in 0..32 {
        runner.step();
        let snap = runner.get_snapshot();
        let meta = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<FractalPassMeta>())
            .copied()
            .unwrap();
        if strides.last().copied() != Some(meta.stride) {
            strides.push(meta.stride);
        }
        if meta.is_final {
            break;
        }
    }
    assert_eq!(strides, vec![8, 4, 2, 1]);
    let expected = render_fractal_color_image_with_formula(&req, Some(&job));
    assert_eq!(
        *runner.get_snapshot().pixels.read().unwrap(),
        expected.pixels
    );
}

#[test]
fn test_custom_fractal_url_roundtrip() {
    let mut pmap = BTreeMap::new();
    pmap.insert("a".to_string(), 1.25);
    let mut viewer = FractalViewer {
        mode: FractalMode::CustomMandelbrot,
        ..Default::default()
    };
    viewer.math_input.set_expression("a * z^3 + c");
    viewer.math_input.set_params_map(&pmap);
    let saved = viewer.save_state().expect("custom fractal state saves");
    assert!(saved.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);
    let mut restored = FractalViewer::default();
    restored
        .try_load_state(&saved)
        .expect("custom fractal state loads");
    assert_eq!(restored.mode, FractalMode::CustomMandelbrot);
    assert_eq!(restored.math_input.raw_text, "a * z^3 + c");
    assert_eq!(restored.math_input.params_map(), pmap);
}

#[test]
fn test_custom_fractal_glsl_shader_export_and_reserved_ident_guard() {
    use crate::framework::ShaderDialect;

    let mut pmap = BTreeMap::new();
    pmap.insert("a".to_string(), 1.25);
    let mut viewer = FractalViewer {
        mode: FractalMode::CustomJulia,
        high_contrast: true,
        ..Default::default()
    };
    viewer.math_input.set_expression("a * z^3 + c");
    viewer.math_input.set_params_map(&pmap);

    let webgl = viewer
        .export_glsl_shader(ShaderDialect::WebGl2Es300)
        .expect("custom Julia WebGL2 shader");
    assert!(webgl.starts_with("#version 300 es"));
    assert!(webgl.contains("const bool IS_JULIA = true;"));
    assert!(webgl.contains("const vec2 a = vec2(1.25, 0.0);"));
    assert!(webgl.contains("step_custom_fractal(z, c)"));

    viewer.mode = FractalMode::CustomMandelbrot;
    let st = viewer
        .export_glsl_shader(ShaderDialect::Shadertoy)
        .expect("custom Mandelbrot Shadertoy shader");
    assert!(st.contains("const bool IS_JULIA = false;"));
    assert!(st.contains("void mainImage("));

    viewer.math_input.set_expression("z^2 + c + MAX_ITER");
    assert!(viewer
        .export_glsl_shader(ShaderDialect::WebGl2Es300)
        .is_err());
}
