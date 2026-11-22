use std::collections::HashMap;

struct StrCursor<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> StrCursor<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            src: s.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.src.get(self.pos).copied()
    }

    fn eat(&mut self, ch: u8) -> bool {
        if self.peek() == Some(ch) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn read_ident(&mut self) -> &'a str {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.src.len()
            && (self.src[self.pos].is_ascii_alphanumeric() || self.src[self.pos] == b'_')
        {
            self.pos += 1;
        }
        std::str::from_utf8(&self.src[start..self.pos]).unwrap()
    }

    fn read_number(&mut self) -> f32 {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            let is_exp_sign = (b == b'+' || b == b'-')
                && self.pos > start
                && matches!(self.src[self.pos - 1], b'e' | b'E');
            if b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E') || is_exp_sign {
                self.pos += 1;
            } else {
                break;
            }
        }
        std::str::from_utf8(&self.src[start..self.pos])
            .unwrap()
            .parse::<f32>()
            .unwrap()
    }
}

pub(super) fn eval_glsl_scalar(src: &str, vars: &HashMap<&str, f32>) -> f32 {
    let mut cur = StrCursor::new(src);
    let val = parse_scalar_add(&mut cur, vars);
    cur.skip_ws();
    assert_eq!(cur.pos, cur.src.len(), "Trailing tokens in `{src}`");
    val
}

fn parse_scalar_add(cur: &mut StrCursor<'_>, vars: &HashMap<&str, f32>) -> f32 {
    let mut acc = parse_scalar_mul(cur, vars);
    loop {
        if cur.eat(b'+') {
            acc += parse_scalar_mul(cur, vars);
        } else if cur.eat(b'-') {
            acc -= parse_scalar_mul(cur, vars);
        } else {
            break;
        }
    }
    acc
}

fn parse_scalar_mul(cur: &mut StrCursor<'_>, vars: &HashMap<&str, f32>) -> f32 {
    let mut acc = parse_scalar_unary(cur, vars);
    while cur.eat(b'*') {
        acc *= parse_scalar_unary(cur, vars);
    }
    acc
}

fn parse_scalar_unary(cur: &mut StrCursor<'_>, vars: &HashMap<&str, f32>) -> f32 {
    if cur.eat(b'-') {
        return -parse_scalar_unary(cur, vars);
    }
    if cur.eat(b'(') {
        let v = parse_scalar_add(cur, vars);
        assert!(cur.eat(b')'));
        return v;
    }
    if cur.peek().is_some_and(|b| b.is_ascii_digit()) {
        return cur.read_number();
    }
    let ident = cur.read_ident();
    if cur.eat(b'(') {
        let a = parse_scalar_add(cur, vars);
        let b = cur.eat(b',').then(|| parse_scalar_add(cur, vars));
        assert!(cur.eat(b')'));
        apply_scalar_fn(ident, a, b)
    } else {
        *vars
            .get(ident)
            .unwrap_or_else(|| panic!("Unbound var {ident}"))
    }
}

fn apply_scalar_fn(name: &str, a: f32, b: Option<f32>) -> f32 {
    match (name, b) {
        ("glsl_div", Some(rhs)) => {
            if rhs.abs() < 1.0e-12 {
                0.0
            } else {
                a / rhs
            }
        }
        ("glsl_pow", Some(rhs)) => sim_glsl_pow(a, rhs),
        ("sin", None) => a.sin(),
        ("cos", None) => a.cos(),
        ("tan", None) => a.tan(),
        ("glsl_exp", None) => a.clamp(-80.0, 80.0).exp(),
        ("glsl_ln", None) => {
            if a <= 1.0e-12 {
                -27.631_021
            } else {
                a.ln()
            }
        }
        ("glsl_log10", None) => {
            if a <= 1.0e-12 {
                -12.0
            } else {
                a.ln() * std::f32::consts::LOG10_E
            }
        }
        ("glsl_sqrt", None) => a.max(0.0).sqrt(),
        ("abs", None) => a.abs(),
        ("glsl_sinh", None) => a.clamp(-80.0, 80.0).sinh(),
        ("glsl_cosh", None) => a.clamp(-80.0, 80.0).cosh(),
        ("tanh", None) => a.tanh(),
        ("glsl_asin", None) => a.clamp(-1.0, 1.0).asin(),
        ("glsl_acos", None) => a.clamp(-1.0, 1.0).acos(),
        ("atan", None) => a.atan(),
        _ => panic!("Unsupported scalar call {name}"),
    }
}

fn sim_glsl_pow(a: f32, b: f32) -> f32 {
    if a.abs() < 1.0e-12 {
        return if b.abs() < 1.0e-12 { 1.0 } else { 0.0 };
    }
    if a > 0.0 {
        return a.powf(b.clamp(-80.0, 80.0));
    }
    let b_round = (b + 0.5).floor();
    if (b - b_round).abs() < 1.0e-6 {
        let mag = (-a).powf(b_round.clamp(-80.0, 80.0));
        return if (b_round.abs() % 2.0) > 0.5 {
            -mag
        } else {
            mag
        };
    }
    0.0
}

pub(super) fn eval_glsl_complex(src: &str, z: [f32; 2]) -> [f32; 2] {
    let mut cur = StrCursor::new(src);
    let val = parse_complex_add(&mut cur, z);
    cur.skip_ws();
    assert_eq!(cur.pos, cur.src.len(), "Trailing tokens in `{src}`");
    val
}

fn parse_complex_add(cur: &mut StrCursor<'_>, z: [f32; 2]) -> [f32; 2] {
    let mut acc = parse_complex_unary(cur, z);
    loop {
        if cur.eat(b'+') {
            let rhs = parse_complex_unary(cur, z);
            acc = [acc[0] + rhs[0], acc[1] + rhs[1]];
        } else if cur.eat(b'-') {
            let rhs = parse_complex_unary(cur, z);
            acc = [acc[0] - rhs[0], acc[1] - rhs[1]];
        } else {
            break;
        }
    }
    acc
}

fn parse_complex_unary(cur: &mut StrCursor<'_>, z: [f32; 2]) -> [f32; 2] {
    if cur.eat(b'-') {
        let v = parse_complex_unary(cur, z);
        return [-v[0], -v[1]];
    }
    if cur.eat(b'(') {
        let v = parse_complex_add(cur, z);
        assert!(cur.eat(b')'));
        return v;
    }
    let ident = cur.read_ident();
    if ident == "z" {
        return z;
    }
    assert!(cur.eat(b'('), "Expected `(` after `{ident}`");
    if ident == "vec2" {
        let mut s_vars = HashMap::new();
        let re = parse_scalar_add(&mut *cur, &s_vars);
        assert!(cur.eat(b','));
        s_vars.clear();
        let im = parse_scalar_add(&mut *cur, &s_vars);
        assert!(cur.eat(b')'));
        return [re, im];
    }
    let a = parse_complex_add(cur, z);
    let b = cur.eat(b',').then(|| parse_complex_add(cur, z));
    assert!(cur.eat(b')'));
    apply_complex_fn(ident, a, b)
}

fn sim_carg(z: [f32; 2]) -> f32 {
    if z[1].abs() == 0.0 {
        if z[0] < 0.0 {
            std::f32::consts::PI
        } else {
            0.0
        }
    } else {
        z[1].atan2(z[0])
    }
}

fn sim_cln(z: [f32; 2]) -> [f32; 2] {
    let r = z[0].hypot(z[1]);
    if r < 1.0e-12 {
        [-27.631_021, 0.0]
    } else {
        [r.ln(), sim_carg(z)]
    }
}

fn sim_cexp(z: [f32; 2]) -> [f32; 2] {
    let ex = z[0].clamp(-80.0, 80.0).exp();
    [ex * z[1].cos(), ex * z[1].sin()]
}

fn sim_cmul(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn sim_cdiv(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let denom = b[0] * b[0] + b[1] * b[1];
    if denom < 1.0e-12 {
        [0.0, 0.0]
    } else {
        [
            (a[0] * b[0] + a[1] * b[1]) / denom,
            (a[1] * b[0] - a[0] * b[1]) / denom,
        ]
    }
}

fn sim_csqrt(z: [f32; 2]) -> [f32; 2] {
    let r = z[0].hypot(z[1]);
    if r == 0.0 {
        return [0.0, 0.0];
    }
    let re = (0.5 * (r + z[0])).max(0.0).sqrt();
    let im = (0.5 * (r - z[0])).max(0.0).sqrt();
    [re, if z[1] < 0.0 { -im } else { im }]
}

fn apply_complex_fn(name: &str, a: [f32; 2], b: Option<[f32; 2]>) -> [f32; 2] {
    match (name, b) {
        ("cmul", Some(rhs)) => sim_cmul(a, rhs),
        ("cdiv", Some(rhs)) => sim_cdiv(a, rhs),
        ("cpow", Some(rhs)) => {
            let r = a[0].hypot(a[1]);
            if r < 1.0e-12 {
                if rhs[0].abs() < 1.0e-12 && rhs[1].abs() < 1.0e-12 {
                    [1.0, 0.0]
                } else {
                    [0.0, 0.0]
                }
            } else {
                sim_cexp(sim_cmul(rhs, [r.ln(), sim_carg(a)]))
            }
        }
        ("cln", None) => sim_cln(a),
        ("clog10", None) => {
            let l = sim_cln(a);
            [
                l[0] * std::f32::consts::LOG10_E,
                l[1] * std::f32::consts::LOG10_E,
            ]
        }
        ("cexp", None) => sim_cexp(a),
        ("csqrt", None) => sim_csqrt(a),
        ("cabs", None) => [a[0].hypot(a[1]), 0.0],
        ("csin", None) => [a[0].sin() * a[1].cosh(), a[0].cos() * a[1].sinh()],
        ("ccos", None) => [a[0].cos() * a[1].cosh(), -a[0].sin() * a[1].sinh()],
        ("ctan", None) => sim_cdiv(
            [a[0].sin() * a[1].cosh(), a[0].cos() * a[1].sinh()],
            [a[0].cos() * a[1].cosh(), -a[0].sin() * a[1].sinh()],
        ),
        ("csinh", None) => [a[0].sinh() * a[1].cos(), a[0].cosh() * a[1].sin()],
        ("ccosh", None) => [a[0].cosh() * a[1].cos(), a[0].sinh() * a[1].sin()],
        ("ctanh", None) => sim_cdiv(
            [a[0].sinh() * a[1].cos(), a[0].cosh() * a[1].sin()],
            [a[0].cosh() * a[1].cos(), a[0].sinh() * a[1].sin()],
        ),
        ("casin", None) => sim_casin(a),
        ("cacos", None) => {
            let s = sim_casin(a);
            [std::f32::consts::FRAC_PI_2 - s[0], -s[1]]
        }
        ("catan", None) => {
            let d1 = sim_cln([1.0 + a[1], -a[0]]);
            let d2 = sim_cln([1.0 - a[1], a[0]]);
            [-0.5 * (d1[1] - d2[1]), 0.5 * (d1[0] - d2[0])]
        }
        _ => panic!("Unsupported complex call {name}"),
    }
}

fn sim_casin(z: [f32; 2]) -> [f32; 2] {
    let iz = [-z[1], z[0]];
    let zz = sim_cmul(z, z);
    let root = sim_csqrt([1.0 - zz[0], -zz[1]]);
    let w = sim_cln([iz[0] + root[0], iz[1] + root[1]]);
    [w[1], -w[0]]
}
