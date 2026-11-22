//! IEEE-safe GLSL ES 3.00 helper preambles for scalar (`float`) and complex (`vec2`) AST evaluation.

/// Standalone GLSL ES 3.00 helper functions guarding scalar `float` division, exponentiation,
/// logarithms (`ln`, `log10`), square roots, hyperbolic functions, and inverse trig domains.
pub const GLSL_SCALAR_HELPERS: &str = r#"float glsl_div(float a, float b) {
    return (abs(b) < 1.0e-12) ? 0.0 : (a / b);
}

float glsl_pow(float a, float b) {
    if (abs(a) < 1.0e-12) {
        return (abs(b) < 1.0e-12) ? 1.0 : 0.0;
    }
    if (a > 0.0) {
        return pow(a, clamp(b, -80.0, 80.0));
    }
    float b_round = floor(b + 0.5);
    if (abs(b - b_round) < 1.0e-6) {
        float mag = pow(-a, clamp(b_round, -80.0, 80.0));
        return (mod(abs(b_round), 2.0) > 0.5) ? -mag : mag;
    }
    return 0.0;
}

float glsl_exp(float x) {
    return exp(clamp(x, -80.0, 80.0));
}

float glsl_ln(float x) {
    return (x <= 1.0e-12) ? -27.631021115928547 : log(x);
}

float glsl_log10(float x) {
    return (x <= 1.0e-12) ? -12.0 : (log(x) * 0.4342944819032518);
}

float glsl_sqrt(float x) {
    return sqrt(max(x, 0.0));
}

float glsl_sinh(float x) {
    return sinh(clamp(x, -80.0, 80.0));
}

float glsl_cosh(float x) {
    return cosh(clamp(x, -80.0, 80.0));
}

float glsl_asin(float x) {
    return asin(clamp(x, -1.0, 1.0));
}

float glsl_acos(float x) {
    return acos(clamp(x, -1.0, 1.0));
}

float glsl_step(float x) {
    return (x < 0.0) ? 0.0 : 1.0;
}

float glsl_rect(float x) {
    return (abs(x) <= 0.5) ? 1.0 : 0.0;
}

float glsl_clamp(float x, float lo, float hi) {
    return (lo > hi) ? 0.0 : clamp(x, lo, hi);
}

float glsl_atan2(float y, float x) {
    if (abs(y) == 0.0) {
        return (x < 0.0) ? 3.141592653589793 : 0.0;
    }
    return atan(y, x);
}
"#;

/// Standalone GLSL ES 3.00 helper functions implementing `vec2` complex arithmetic,
/// division-by-zero guards (`dot(b, b) < 1e-12`), principal branch-cut `cln`/`cpow`/`csqrt`,
/// `clog10`, and all 20 built-in [`super::Func`] variants.
pub const GLSL_COMPLEX_HELPERS: &str = r#"vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 cdiv(vec2 a, vec2 b) {
    float denom = dot(b, b);
    if (denom < 1.0e-12) {
        return vec2(0.0, 0.0);
    }
    return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / denom;
}

float carg(vec2 z) {
    if (abs(z.y) == 0.0) {
        return (z.x < 0.0) ? 3.141592653589793 : 0.0;
    }
    return atan(z.y, z.x);
}

vec2 cln(vec2 z) {
    float r = length(z);
    if (r < 1.0e-12) {
        return vec2(-27.631021115928547, 0.0);
    }
    return vec2(log(r), carg(z));
}

vec2 clog10(vec2 z) {
    return cln(z) * 0.4342944819032518;
}

vec2 cexp(vec2 z) {
    float ex = exp(clamp(z.x, -80.0, 80.0));
    return vec2(ex * cos(z.y), ex * sin(z.y));
}

vec2 cpow(vec2 a, vec2 b) {
    float r = length(a);
    if (r < 1.0e-12) {
        return (abs(b.x) < 1.0e-12 && abs(b.y) < 1.0e-12)
            ? vec2(1.0, 0.0)
            : vec2(0.0, 0.0);
    }
    return cexp(cmul(b, vec2(log(r), carg(a))));
}

vec2 csqrt(vec2 z) {
    float r = length(z);
    if (r == 0.0) {
        return vec2(0.0, 0.0);
    }
    float re = sqrt(max(0.5 * (r + z.x), 0.0));
    float im = sqrt(max(0.5 * (r - z.x), 0.0));
    return vec2(re, (z.y < 0.0) ? -im : im);
}

vec2 cabs(vec2 z) {
    return vec2(length(z), 0.0);
}

vec2 csin(vec2 z) {
    float cy = clamp(z.y, -80.0, 80.0);
    return vec2(sin(z.x) * cosh(cy), cos(z.x) * sinh(cy));
}

vec2 ccos(vec2 z) {
    float cy = clamp(z.y, -80.0, 80.0);
    return vec2(cos(z.x) * cosh(cy), -sin(z.x) * sinh(cy));
}

vec2 ctan(vec2 z) {
    return cdiv(csin(z), ccos(z));
}

vec2 csinh(vec2 z) {
    float cx = clamp(z.x, -80.0, 80.0);
    return vec2(sinh(cx) * cos(z.y), cosh(cx) * sin(z.y));
}

vec2 ccosh(vec2 z) {
    float cx = clamp(z.x, -80.0, 80.0);
    return vec2(cosh(cx) * cos(z.y), sinh(cx) * sin(z.y));
}

vec2 ctanh(vec2 z) {
    return cdiv(csinh(z), ccosh(z));
}

vec2 casin(vec2 z) {
    vec2 iz = vec2(-z.y, z.x);
    vec2 root = csqrt(vec2(1.0, 0.0) - cmul(z, z));
    vec2 w = cln(iz + root);
    return vec2(w.y, -w.x);
}

vec2 cacos(vec2 z) {
    vec2 a = casin(z);
    return vec2(1.5707963267948966 - a.x, -a.y);
}

vec2 catan(vec2 z) {
    vec2 one_minus_iz = vec2(1.0 + z.y, -z.x);
    vec2 one_plus_iz = vec2(1.0 - z.y, z.x);
    vec2 d = cln(one_minus_iz) - cln(one_plus_iz);
    return vec2(-0.5 * d.y, 0.5 * d.x);
}

vec2 cstep(vec2 z) {
    return vec2((z.x < 0.0) ? 0.0 : 1.0, 0.0);
}

vec2 crect(vec2 z) {
    return vec2((abs(z.x) <= 0.5) ? 1.0 : 0.0, 0.0);
}

vec2 cmin(vec2 a, vec2 b) {
    return vec2(min(a.x, b.x), 0.0);
}

vec2 cmax(vec2 a, vec2 b) {
    return vec2(max(a.x, b.x), 0.0);
}

vec2 cclamp(vec2 x, vec2 lo, vec2 hi) {
    return vec2((lo.x > hi.x) ? 0.0 : clamp(x.x, lo.x, hi.x), 0.0);
}

vec2 catan2(vec2 y, vec2 x) {
    return vec2(carg(vec2(x.x, y.x)), 0.0);
}
"#;

const GLSL_KEYWORDS: &[&str] = &[
    "attribute",
    "bool",
    "break",
    "bvec2",
    "bvec3",
    "bvec4",
    "case",
    "centroid",
    "const",
    "continue",
    "default",
    "discard",
    "do",
    "else",
    "false",
    "flat",
    "float",
    "for",
    "highp",
    "if",
    "in",
    "inout",
    "int",
    "invariant",
    "ivec2",
    "ivec3",
    "ivec4",
    "layout",
    "lowp",
    "main",
    "mat2",
    "mat3",
    "mat4",
    "mediump",
    "out",
    "precision",
    "return",
    "smooth",
    "struct",
    "switch",
    "true",
    "uint",
    "uniform",
    "uvec2",
    "uvec3",
    "uvec4",
    "varying",
    "vec2",
    "vec3",
    "vec4",
    "void",
    "while",
];

pub(super) fn is_valid_glsl_ident(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return false;
    }
    if name.starts_with("gl_") || name.contains("__") {
        return false;
    }
    GLSL_KEYWORDS.binary_search(&name).is_err()
}
