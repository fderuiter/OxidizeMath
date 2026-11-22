//! Standalone WGSL and GLSL helper preludes for real and complex AST shader lowering.

/// Target shading language for AST expression transpilation and prelude emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderTarget {
    /// WebGPU Shading Language (`WGSL`, WebGPU / Bevy).
    Wgsl,
    /// OpenGL ES Shading Language 3.00 (`GLSL`, WebGL2 / Shadertoy).
    Glsl,
}

/// Returns the WGSL helper prelude for real scalar (`f32`) lowered AST expressions.
#[must_use]
pub const fn wgsl_real_prelude() -> &'static str {
    r#"fn om_safe_div(lhs: f32, rhs: f32) -> f32 {
    if (abs(rhs) < 1.0e-12) {
        return 0.0;
    }
    return lhs / rhs;
}

fn om_safe_pow(base: f32, exponent: f32) -> f32 {
    if (abs(exponent) < 1.0e-12) {
        return 1.0;
    }
    if (abs(base) < 1.0e-30) {
        return 0.0;
    }
    if (base > 0.0) {
        return pow(base, clamp(exponent, -80.0, 80.0));
    }
    let rounded = round(exponent);
    if (abs(exponent - rounded) < 1.0e-5) {
        let mag = pow(abs(base), clamp(rounded, -80.0, 80.0));
        let parity = abs(fract(rounded * 0.5) * 2.0);
        return select(mag, -mag, parity > 0.5);
    }
    return 0.0;
}

fn om_ln(x: f32) -> f32 {
    return log(max(x, 1.0e-30));
}

fn om_log10(x: f32) -> f32 {
    return log(max(x, 1.0e-30)) * 0.43429448;
}
"#
}

/// Returns the GLSL ES 3.00 helper prelude for real scalar (`float`) lowered AST expressions.
#[must_use]
pub const fn glsl_real_prelude() -> &'static str {
    r#"float om_safe_div(float lhs, float rhs) {
    if (abs(rhs) < 1.0e-12) {
        return 0.0;
    }
    return lhs / rhs;
}

float om_safe_pow(float base, float exponent) {
    if (abs(exponent) < 1.0e-12) {
        return 1.0;
    }
    if (abs(base) < 1.0e-30) {
        return 0.0;
    }
    if (base > 0.0) {
        return pow(base, clamp(exponent, -80.0, 80.0));
    }
    float rounded = round(exponent);
    if (abs(exponent - rounded) < 1.0e-5) {
        float mag = pow(abs(base), clamp(rounded, -80.0, 80.0));
        float parity = abs(fract(rounded * 0.5) * 2.0);
        return (parity > 0.5) ? -mag : mag;
    }
    return 0.0;
}

float om_ln(float x) {
    return log(max(x, 1.0e-30));
}

float om_log10(float x) {
    return log(max(x, 1.0e-30)) * 0.43429448;
}
"#
}

const WGSL_COMPLEX_PRELUDE: &str = r#"fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn c_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denom = dot(b, b);
    if (denom < 1.0e-12) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

fn c_exp(z: vec2<f32>) -> vec2<f32> {
    let ex = exp(clamp(z.x, -80.0, 80.0));
    return vec2<f32>(ex * cos(z.y), ex * sin(z.y));
}

fn c_ln(z: vec2<f32>) -> vec2<f32> {
    let r = max(length(z), 1.0e-30);
    return vec2<f32>(log(r), atan2(z.y, z.x));
}

fn c_log10(z: vec2<f32>) -> vec2<f32> {
    return c_ln(z) * 0.43429448;
}

fn c_sqrt(z: vec2<f32>) -> vec2<f32> {
    let r = length(z);
    if (r == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let u = sqrt(max(0.5 * (r + abs(z.x)), 0.0));
    let v = sqrt(max(0.5 * (r - abs(z.x)), 0.0));
    if (z.x >= 0.0) {
        return vec2<f32>(u, select(-v, v, z.y >= 0.0));
    }
    return vec2<f32>(v, select(-u, u, z.y >= 0.0));
}

fn c_abs(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(length(z), 0.0);
}

fn c_pow(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (abs(b.y) < 1.0e-7) {
        if (abs(b.x) < 1.0e-7) {
            return vec2<f32>(1.0, 0.0);
        }
        if (abs(b.x - 1.0) < 1.0e-7) {
            return a;
        }
        if (abs(b.x - 2.0) < 1.0e-7) {
            return c_mul(a, a);
        }
        if (abs(b.x - 3.0) < 1.0e-7) {
            return c_mul(c_mul(a, a), a);
        }
        if (abs(b.x - 4.0) < 1.0e-7) {
            let a2 = c_mul(a, a);
            return c_mul(a2, a2);
        }
        if (abs(b.x + 1.0) < 1.0e-7) {
            return c_div(vec2<f32>(1.0, 0.0), a);
        }
    }
    if (dot(a, a) < 1.0e-24) {
        return vec2<f32>(0.0, 0.0);
    }
    return c_exp(c_mul(b, c_ln(a)));
}

fn c_sin(z: vec2<f32>) -> vec2<f32> {
    let y = clamp(z.y, -80.0, 80.0);
    return vec2<f32>(sin(z.x) * cosh(y), cos(z.x) * sinh(y));
}

fn c_cos(z: vec2<f32>) -> vec2<f32> {
    let y = clamp(z.y, -80.0, 80.0);
    return vec2<f32>(cos(z.x) * cosh(y), -sin(z.x) * sinh(y));
}

fn c_tan(z: vec2<f32>) -> vec2<f32> {
    return c_div(c_sin(z), c_cos(z));
}

fn c_sinh(z: vec2<f32>) -> vec2<f32> {
    let x = clamp(z.x, -80.0, 80.0);
    return vec2<f32>(sinh(x) * cos(z.y), cosh(x) * sin(z.y));
}

fn c_cosh(z: vec2<f32>) -> vec2<f32> {
    let x = clamp(z.x, -80.0, 80.0);
    return vec2<f32>(cosh(x) * cos(z.y), sinh(x) * sin(z.y));
}

fn c_tanh(z: vec2<f32>) -> vec2<f32> {
    return c_div(c_sinh(z), c_cosh(z));
}

fn c_asin(z: vec2<f32>) -> vec2<f32> {
    let iz = vec2<f32>(-z.y, z.x);
    let root = c_sqrt(vec2<f32>(1.0, 0.0) - c_mul(z, z));
    let w = c_ln(iz + root);
    return vec2<f32>(w.y, -w.x);
}

fn c_acos(z: vec2<f32>) -> vec2<f32> {
    let root = c_sqrt(vec2<f32>(1.0, 0.0) - c_mul(z, z));
    let i_root = vec2<f32>(-root.y, root.x);
    let w = c_ln(z + i_root);
    return vec2<f32>(w.y, -w.x);
}

fn c_atan(z: vec2<f32>) -> vec2<f32> {
    let iz = vec2<f32>(-z.y, z.x);
    let d = c_ln(vec2<f32>(1.0, 0.0) - iz) - c_ln(vec2<f32>(1.0, 0.0) + iz);
    return vec2<f32>(-0.5 * d.y, 0.5 * d.x);
}
"#;

/// Returns the WGSL helper prelude for `vec2<f32>` complex arithmetic and all 14 elementary functions.
#[must_use]
pub const fn wgsl_complex_prelude() -> &'static str {
    WGSL_COMPLEX_PRELUDE
}

const GLSL_COMPLEX_PRELUDE: &str = r#"vec2 c_mul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 c_div(vec2 a, vec2 b) {
    float denom = dot(b, b);
    if (denom < 1.0e-12) {
        return vec2(0.0, 0.0);
    }
    return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

vec2 c_exp(vec2 z) {
    float ex = exp(clamp(z.x, -80.0, 80.0));
    return vec2(ex * cos(z.y), ex * sin(z.y));
}

vec2 c_ln(vec2 z) {
    float r = max(length(z), 1.0e-30);
    return vec2(log(r), atan(z.y, z.x));
}

vec2 c_log10(vec2 z) {
    return c_ln(z) * 0.43429448;
}

vec2 c_sqrt(vec2 z) {
    float r = length(z);
    if (r == 0.0) {
        return vec2(0.0, 0.0);
    }
    float u = sqrt(max(0.5 * (r + abs(z.x)), 0.0));
    float v = sqrt(max(0.5 * (r - abs(z.x)), 0.0));
    if (z.x >= 0.0) {
        return vec2(u, (z.y >= 0.0) ? v : -v);
    }
    return vec2(v, (z.y >= 0.0) ? u : -u);
}

vec2 c_abs(vec2 z) {
    return vec2(length(z), 0.0);
}

vec2 c_pow(vec2 a, vec2 b) {
    if (abs(b.y) < 1.0e-7) {
        if (abs(b.x) < 1.0e-7) {
            return vec2(1.0, 0.0);
        }
        if (abs(b.x - 1.0) < 1.0e-7) {
            return a;
        }
        if (abs(b.x - 2.0) < 1.0e-7) {
            return c_mul(a, a);
        }
        if (abs(b.x - 3.0) < 1.0e-7) {
            return c_mul(c_mul(a, a), a);
        }
        if (abs(b.x - 4.0) < 1.0e-7) {
            vec2 a2 = c_mul(a, a);
            return c_mul(a2, a2);
        }
        if (abs(b.x + 1.0) < 1.0e-7) {
            return c_div(vec2(1.0, 0.0), a);
        }
    }
    if (dot(a, a) < 1.0e-24) {
        return vec2(0.0, 0.0);
    }
    return c_exp(c_mul(b, c_ln(a)));
}

vec2 c_sin(vec2 z) {
    float y = clamp(z.y, -80.0, 80.0);
    return vec2(sin(z.x) * cosh(y), cos(z.x) * sinh(y));
}

vec2 c_cos(vec2 z) {
    float y = clamp(z.y, -80.0, 80.0);
    return vec2(cos(z.x) * cosh(y), -sin(z.x) * sinh(y));
}

vec2 c_tan(vec2 z) {
    return c_div(c_sin(z), c_cos(z));
}

vec2 c_sinh(vec2 z) {
    float x = clamp(z.x, -80.0, 80.0);
    return vec2(sinh(x) * cos(z.y), cosh(x) * sin(z.y));
}

vec2 c_cosh(vec2 z) {
    float x = clamp(z.x, -80.0, 80.0);
    return vec2(cosh(x) * cos(z.y), sinh(x) * sin(z.y));
}

vec2 c_tanh(vec2 z) {
    return c_div(c_sinh(z), c_cosh(z));
}

vec2 c_asin(vec2 z) {
    vec2 iz = vec2(-z.y, z.x);
    vec2 root = c_sqrt(vec2(1.0, 0.0) - c_mul(z, z));
    vec2 w = c_ln(iz + root);
    return vec2(w.y, -w.x);
}

vec2 c_acos(vec2 z) {
    vec2 root = c_sqrt(vec2(1.0, 0.0) - c_mul(z, z));
    vec2 i_root = vec2(-root.y, root.x);
    vec2 w = c_ln(z + i_root);
    return vec2(w.y, -w.x);
}

vec2 c_atan(vec2 z) {
    vec2 iz = vec2(-z.y, z.x);
    vec2 d = c_ln(vec2(1.0, 0.0) - iz) - c_ln(vec2(1.0, 0.0) + iz);
    return vec2(-0.5 * d.y, 0.5 * d.x);
}
"#;

/// Returns the GLSL ES 3.00 helper prelude for `vec2` complex arithmetic and all 14 elementary functions.
#[must_use]
pub const fn glsl_complex_prelude() -> &'static str {
    GLSL_COMPLEX_PRELUDE
}

/// Returns the complex `vec2` helper prelude for `target`.
#[must_use]
pub const fn complex_prelude_for(target: ShaderTarget) -> &'static str {
    match target {
        ShaderTarget::Wgsl => wgsl_complex_prelude(),
        ShaderTarget::Glsl => glsl_complex_prelude(),
    }
}

/// Returns the scalar `f32`/`float` helper prelude for `target`.
#[must_use]
pub const fn real_prelude_for(target: ShaderTarget) -> &'static str {
    match target {
        ShaderTarget::Wgsl => wgsl_real_prelude(),
        ShaderTarget::Glsl => glsl_real_prelude(),
    }
}
