//! Headless CLI entrypoint for `math_explorer` supporting global sensitivity sweeps
//! (`sensitivity`) and empirical CSV curve calibration (`calibrate`).

#![cfg_attr(any(), verified(opt_out = "cli_binary"))]

use math_explorer::ast::{CompiledExpr, parse};
use math_explorer::prelude::{
    MAX_FIT_PARAMS, ParameterUncertainty, SensitivityEngine, TornadoBounds,
    levenberg_marquardt_numeric, sum_squared_errors_numeric,
};
use std::fs;
use std::process::ExitCode;

#[derive(Debug, Clone)]
struct ParamArg {
    name: String,
    baseline: f64,
    min: f64,
    max: f64,
    sigma: f64,
}

fn parse_param_spec(spec: &str) -> Result<ParamArg, String> {
    let (name, rest) = spec
        .split_once('=')
        .ok_or_else(|| format!("Invalid --param `{spec}`; expected name=val[:min:max[:sigma]]"))?;
    let name = name.trim();
    if name.is_empty() {
        return Err("Parameter name cannot be empty".to_string());
    }
    let parts: Vec<&str> = rest.split(':').collect();
    let baseline: f64 = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("Invalid numeric value in `{spec}`"))?;
    let span = baseline.abs().max(1.0) * 0.25;
    let min = if parts.len() >= 3 {
        parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid min bound in `{spec}`"))?
    } else {
        baseline - span
    };
    let max = if parts.len() >= 3 {
        parts[2]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid max bound in `{spec}`"))?
    } else {
        baseline + span
    };
    let sigma = if parts.len() >= 4 {
        parts[3]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid sigma in `{spec}`"))?
    } else {
        (max - min).abs() * 0.1
    };
    if !baseline.is_finite() || !min.is_finite() || !max.is_finite() || !sigma.is_finite() {
        return Err(format!("Non-finite parameter values in `{spec}`"));
    }
    Ok(ParamArg {
        name: name.to_string(),
        baseline,
        min: min.min(max),
        max: min.max(max),
        sigma: sigma.abs(),
    })
}

fn default_sensitivity_params() -> Vec<ParamArg> {
    vec![
        ParamArg {
            name: "dose".to_string(),
            baseline: 100.0,
            min: 50.0,
            max: 200.0,
            sigma: 10.0,
        },
        ParamArg {
            name: "ka".to_string(),
            baseline: 1.2,
            min: 0.4,
            max: 3.0,
            sigma: 0.15,
        },
        ParamArg {
            name: "ke".to_string(),
            baseline: 0.12,
            min: 0.03,
            max: 0.4,
            sigma: 0.02,
        },
        ParamArg {
            name: "volume".to_string(),
            baseline: 40.0,
            min: 20.0,
            max: 80.0,
            sigma: 4.0,
        },
    ]
}

fn execute_sensitivity_sweeps(
    expr_str: &str,
    params: &[ParamArg],
    samples: usize,
    seed: u64,
) -> Result<String, String> {
    let expr = parse(expr_str).map_err(|e| format!("Expression parse error: {e}"))?;
    let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
    let compiled = CompiledExpr::compile_with_vars_and_params(&expr, &[], &param_names)
        .map_err(|e| format!("Expression compile error: {e}"))?;

    let pairs: Vec<(&str, f64)> = params
        .iter()
        .map(|p| (p.name.as_str(), p.baseline))
        .collect();
    let bounds: Vec<TornadoBounds> = params
        .iter()
        .map(|p| TornadoBounds::new(p.min, p.max))
        .collect();
    let unc: Vec<ParameterUncertainty> = params
        .iter()
        .map(|p| ParameterUncertainty::new(p.sigma, p.min, p.max))
        .collect();

    let mut engine = SensitivityEngine::new();
    while !engine.is_converged() {
        engine.compute_slice(&pairs, |vals| compiled.eval_slice(vals));
    }
    while !engine.is_tornado_converged() {
        engine.step_tornado_slice(&pairs, &bounds, 8, |vals| compiled.eval_slice(vals));
    }
    while !engine.is_monte_carlo_converged() {
        engine.step_monte_carlo_slice(&pairs, &unc, seed, samples, 8, |vals| {
            compiled.eval_slice(vals)
        });
    }
    Ok(format_sensitivity_report(expr_str, &engine))
}

fn run_sensitivity_subcommand(args: &[String]) -> Result<String, String> {
    let mut expr_str = "(dose / volume) * (ka / (ka + ke))".to_string();
    let mut params = Vec::new();
    let mut samples: usize = 128;
    let mut seed: u64 = 42;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--expr" => {
                idx += 1;
                expr_str = args.get(idx).cloned().ok_or("Missing value for --expr")?;
            }
            "--param" => {
                idx += 1;
                let raw = args.get(idx).ok_or("Missing value for --param")?;
                params.push(parse_param_spec(raw)?);
            }
            "--samples" => {
                idx += 1;
                samples = args
                    .get(idx)
                    .ok_or("Missing value for --samples")?
                    .parse()
                    .map_err(|_| "Invalid integer for --samples")?;
            }
            "--seed" => {
                idx += 1;
                seed = args
                    .get(idx)
                    .ok_or("Missing value for --seed")?
                    .parse()
                    .map_err(|_| "Invalid u64 for --seed")?;
            }
            other => return Err(format!("Unknown sensitivity argument `{other}`")),
        }
        idx += 1;
    }

    if params.is_empty() {
        params = default_sensitivity_params();
    }
    execute_sensitivity_sweeps(&expr_str, &params, samples, seed)
}

fn format_sensitivity_report(expr_str: &str, engine: &SensitivityEngine) -> String {
    let mut out = format!(
        "Model: {expr_str}\nBaseline: {:.6}\nparameter,baseline,dy_dp,elasticity,tornado_min,tornado_max,swing\n",
        engine.baseline_output
    );
    for (m, iv) in engine.metrics.iter().zip(engine.tornado_intervals()) {
        out.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            m.param_name,
            m.baseline_value,
            m.partial_derivative,
            m.elasticity,
            iv.min_output,
            iv.max_output,
            iv.swing
        ));
    }
    if let Some(mc) = engine.monte_carlo_summary() {
        out.push_str(&format!(
            "MonteCarlo(n={}): mean={:.6}, std={:.6}, p05={:.6}, p50={:.6}, p95={:.6}\n",
            mc.sample_count, mc.mean, mc.std_dev, mc.p05, mc.p50, mc.p95
        ));
    }
    out
}

fn parse_csv_series(raw: &str) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for line in raw.lines().flat_map(|l| l.split(';')) {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut cols = trimmed.split(',');
        let (Some(c0), Some(c1)) = (cols.next(), cols.next()) else {
            continue;
        };
        if let (Ok(x), Ok(y)) = (c0.trim().parse::<f64>(), c1.trim().parse::<f64>())
            && x.is_finite()
            && y.is_finite()
        {
            xs.push(x);
            ys.push(y);
        }
    }
    if xs.len() < 2 {
        return Err("Calibration requires at least 2 finite (x, y) observations".to_string());
    }
    Ok((xs, ys))
}

fn run_calibrate_subcommand(args: &[String]) -> Result<String, String> {
    let mut expr_str = "a * exp(-k * x)".to_string();
    let mut params = Vec::new();
    let mut csv_path: Option<String> = None;
    let mut inline_points: Option<String> = None;

    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--expr" => {
                idx += 1;
                expr_str = args.get(idx).cloned().ok_or("Missing value for --expr")?;
            }
            "--param" => {
                idx += 1;
                let raw = args.get(idx).ok_or("Missing value for --param")?;
                params.push(parse_param_spec(raw)?);
            }
            "--csv" => {
                idx += 1;
                csv_path = Some(args.get(idx).cloned().ok_or("Missing value for --csv")?);
            }
            "--points" => {
                idx += 1;
                inline_points = Some(args.get(idx).cloned().ok_or("Missing value for --points")?);
            }
            other => return Err(format!("Unknown calibrate argument `{other}`")),
        }
        idx += 1;
    }

    if params.is_empty() {
        params.push(parse_param_spec("a=2.0:0.1:50.0")?);
        params.push(parse_param_spec("k=0.25:0.001:5.0")?);
    }
    if params.len() > MAX_FIT_PARAMS {
        return Err(format!(
            "At most {MAX_FIT_PARAMS} parameters supported by levenberg_marquardt_numeric"
        ));
    }

    let raw_csv = if let Some(path) = csv_path {
        fs::read_to_string(&path).map_err(|e| format!("Failed to read `{path}`: {e}"))?
    } else if let Some(inline) = inline_points {
        inline
    } else {
        "0.0,5.0;1.0,3.03265;2.0,1.83940;3.0,1.11565;4.0,0.67668".to_string()
    };
    let (xs, ys) = parse_csv_series(&raw_csv)?;
    fit_compiled_csv_model(&expr_str, &params, &xs, &ys)
}

fn encode_bounded(val: f64, min: f64, max: f64) -> f64 {
    let span = (max - min).max(1e-9);
    let t = ((val - min) / span).clamp(1e-6, 1.0 - 1e-6);
    (t / (1.0 - t)).ln()
}

fn decode_bounded(u: f64, min: f64, max: f64) -> f64 {
    let sig = 1.0 / (1.0 + (-u.clamp(-40.0, 40.0)).exp());
    (min + (max - min) * sig).clamp(min, max)
}

fn fit_compiled_csv_model(
    expr_str: &str,
    params: &[ParamArg],
    xs: &[f64],
    ys: &[f64],
) -> Result<String, String> {
    let expr = parse(expr_str).map_err(|e| format!("Expression parse error: {e}"))?;
    let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
    let compiled = CompiledExpr::compile_with_vars_and_params(&expr, &["x"], &param_names)
        .map_err(|e| format!("Expression compile error: {e}"))?;

    let initial_u: Vec<f64> = params
        .iter()
        .map(|p| encode_bounded(p.baseline, p.min, p.max))
        .collect();
    let series = (xs, ys);
    let raw_u = levenberg_marquardt_numeric(&series, &initial_u, |x, u_slice| {
        let mut decoded = [0.0_f64; MAX_FIT_PARAMS];
        for (i, p) in params.iter().enumerate() {
            decoded[i] = decode_bounded(u_slice[i], p.min, p.max);
        }
        let val = compiled.eval_1d(x, &decoded[..params.len()]);
        val.is_finite().then_some(val)
    })
    .map_err(|e| format!("Calibration failed: {e}"))?;

    let mut fitted = [0.0_f64; MAX_FIT_PARAMS];
    for (i, p) in params.iter().enumerate() {
        fitted[i] = decode_bounded(raw_u[i], p.min, p.max);
    }
    let k = params.len();
    let sse = sum_squared_errors_numeric(&series, &fitted[..k], |x, th| {
        let v = compiled.eval_1d(x, th);
        v.is_finite().then_some(v)
    })
    .unwrap_or(f64::NAN);

    let mut out = format!("Calibrated `{expr_str}` (sse={sse:.8}):\n");
    for (i, p) in params.iter().enumerate() {
        out.push_str(&format!("  {} = {:.6}\n", p.name, fitted[i]));
    }
    Ok(out)
}

fn usage_text() -> &'static str {
    "Usage: math_explorer <subcommand> [options]\n\
     Subcommands:\n\
       sensitivity [--expr <formula>] [--param name=val:min:max:sigma]... [--samples N] [--seed S]\n\
       calibrate   [--expr <formula>] [--param name=seed:min:max]... [--csv <file> | --points <x,y;...>]\n"
}

fn run_cli(args: &[String]) -> Result<String, String> {
    let Some(cmd) = args.first().map(String::as_str) else {
        return Ok(usage_text().to_string());
    };
    match cmd {
        "sensitivity" => run_sensitivity_subcommand(&args[1..]),
        "calibrate" => run_calibrate_subcommand(&args[1..]),
        "--help" | "-h" | "help" => Ok(usage_text().to_string()),
        other => Err(format!("Unknown subcommand `{other}`\n{}", usage_text())),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run_cli(&args) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_sensitivity_subcommand_runs_tornado_and_monte_carlo() {
        let args = vec![
            "sensitivity".to_string(),
            "--samples".to_string(),
            "64".to_string(),
            "--seed".to_string(),
            "123".to_string(),
        ];
        let out = run_cli(&args).expect("sensitivity succeeds");
        assert!(out.contains("Baseline:"));
        assert!(out.contains("dose,100.000000,0.022727,1.000000"));
        assert!(out.contains("MonteCarlo(n=64):"));
    }

    #[test]
    fn cli_calibrate_subcommand_recovers_exponential_rate() {
        let args = vec![
            "calibrate".to_string(),
            "--expr".to_string(),
            "a * exp(-k * x)".to_string(),
            "--param".to_string(),
            "a=2.0:0.1:20.0".to_string(),
            "--param".to_string(),
            "k=0.2:0.01:2.0".to_string(),
            "--points".to_string(),
            "0,5.0;1,3.032653;2,1.839397;3,1.115651;4,0.676676".to_string(),
        ];
        let out = run_cli(&args).expect("calibrate succeeds");
        assert!(out.contains("a = 5.000"));
        assert!(out.contains("k = 0.500"));
    }
}
