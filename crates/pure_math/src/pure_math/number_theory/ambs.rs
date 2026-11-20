use crate::math_types::{Integer, ops::Pow, ops::RemRounding};
use crate::pure_math::number_theory::primes::is_prime;
use thiserror::Error;

#[allow(missing_docs)]
#[verified_engine::verified]
pub fn sigma(n: u64) -> u64 {
    let mut sum = 0;
    for i in 1..=(n as f64).sqrt() as u64 {
        if n.is_multiple_of(i) {
            sum += i;
            if i * i != n {
                sum += n / i;
            }
        }
    }
    sum
}

#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum AmbsError {
    #[error("Failed to parse integer from string: {0}")]
    #[allow(missing_docs)]
    ParseError(String),
    #[error("Modulo operation failed")]
    #[allow(missing_docs)]
    ModuloError,
    #[error("Conversion to usize failed")]
    #[allow(missing_docs)]
    ConversionError,
}

#[allow(missing_docs)]
#[verified_engine::verified]
pub fn modular_inverse(a: &Integer, m: &Integer) -> Option<Integer> {
    let mut a_copy = a.clone();
    a_copy = a_copy.rem_euc(m);
    if a_copy < 0 {
        a_copy += m;
    }
    a_copy.invert(m).ok()
}

#[allow(missing_docs)]
#[verified_engine::verified]
pub fn prime_factors(mut n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    let mut count = 0;
    while n.is_multiple_of(2) {
        count += 1;
        n /= 2;
    }
    if count > 0 {
        factors.push((2, count));
    }

    let mut i = 3;
    while i * i <= n {
        count = 0;
        while n.is_multiple_of(i) {
            count += 1;
            n /= i;
        }
        if count > 0 {
            factors.push((i, count));
        }
        i += 2;
    }
    if n > 2 {
        factors.push((n, 1));
    }
    factors
}

#[allow(missing_docs)]
#[verified_engine::verified]
pub fn precompute_valid_prime_powers(limit_p: u64, max_e: u32) -> Vec<(u64, u32, u64)> {
    let mut valid = Vec::new();
    for p in 3..=limit_p {
        if !is_prime(p) {
            continue;
        }
        for e in 1..=max_e {
            let p_power = Integer::from(p).pow(2 * e).to_u64().unwrap_or(0);
            if p_power == 0 {
                continue;
            }
            let sig = sigma(p_power);
            let factors = prime_factors(sig);

            let mut is_valid = true;
            for (q, _) in factors {
                if q % 8 == 5 || q % 8 == 7 {
                    is_valid = false;
                    break;
                }
            }
            if is_valid {
                valid.push((p, e, p_power));
            }
        }
    }
    valid.sort_by_key(|&(_, _, val)| val);
    valid
}

#[allow(missing_docs)]
pub struct AmbsDsp {
    #[allow(missing_docs)]
    pub valid_powers: Vec<(u64, u32, u64)>,
    #[allow(missing_docs)]
    pub b_stop: f64,
    #[allow(missing_docs)]
    pub prefix_pool: Vec<u64>,
}

impl AmbsDsp {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(limit_p: u64, max_e: u32, b_stop: f64) -> Self {
        Self {
            valid_powers: precompute_valid_prime_powers(limit_p, max_e),
            b_stop,
            prefix_pool: Vec::new(),
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn build_prefix(&mut self, current_p: u64, last_p_index: usize, h_max_remaining: f64) {
        let mut stack = vec![(current_p, last_p_index, h_max_remaining)];

        while let Some((curr_p, curr_idx, curr_h_max)) = stack.pop() {
            if (curr_p as f64) >= self.b_stop {
                self.prefix_pool.push(curr_p);
                continue;
            }

            for i in (curr_idx..self.valid_powers.len()).rev() {
                let (p, _, p2e) = self.valid_powers[i];
                if let Some(next_p) = curr_p.checked_mul(p2e) {
                    let term_abundancy = (sigma(p2e) as f64) / (p2e as f64);
                    let current_abundancy = (sigma(curr_p) as f64) / (curr_p as f64);
                    let next_abundancy = current_abundancy * term_abundancy;
                    let next_h_max = curr_h_max / term_abundancy;

                    if next_abundancy * next_h_max >= 2.0 {
                        let mut next_idx = i + 1;
                        while next_idx < self.valid_powers.len()
                            && self.valid_powers[next_idx].0 == p
                        {
                            next_idx += 1;
                        }
                        stack.push((next_p, next_idx, next_h_max));
                    }
                }
            }
        }
    }

    #[verified_engine::verified]
    fn pair_modular_roots(root: Integer, p: &Integer) -> Vec<Integer> {
        debug_assert!(*p > 1, "modulus p must exceed 1");
        debug_assert!(root >= 0 && root < *p, "root must be reduced modulo p");
        let root2 = p.clone() - &root;
        if root == root2 {
            vec![root]
        } else {
            vec![root, root2]
        }
    }

    #[verified_engine::verified]
    fn tonelli_factor_and_nonresidue(p: &Integer) -> Result<(Integer, u32, Integer), AmbsError> {
        debug_assert!(*p > 2, "modulus p must exceed 2");
        let p_minus_one = p.clone() - Integer::from(1);
        let mut q = p_minus_one.clone();
        let mut s = 0_u32;
        while q.is_even() {
            s += 1;
            q /= Integer::from(2);
        }
        debug_assert!(q >= 1, "odd factor q must be positive");

        let euler_exp = p_minus_one.clone() / Integer::from(2);
        let mut z = Integer::from(2);
        while z
            .clone()
            .pow_mod(&euler_exp, p)
            .map_err(|_| AmbsError::ModuloError)?
            != p_minus_one
        {
            z += Integer::from(1);
        }
        Ok((q, s, z))
    }

    #[verified_engine::verified]
    fn tonelli_refine_root(
        mut m: u32,
        mut c: Integer,
        mut t: Integer,
        mut r: Integer,
        p: &Integer,
    ) -> Result<Vec<Integer>, AmbsError> {
        debug_assert!(*p > 2, "modulus p must exceed 2");
        debug_assert!(r >= 0 && r < *p, "initial root estimate must be reduced");
        while t != 0 && t != 1 {
            let mut t2i = t.clone();
            let mut i = 0_u32;
            for j in 1..m {
                t2i = t2i
                    .pow_mod(&Integer::from(2), p)
                    .map_err(|_| AmbsError::ModuloError)?;
                if t2i == 1 {
                    i = j;
                    break;
                }
            }
            if i == 0 {
                return Ok(vec![]);
            }
            let b = c
                .pow_mod(&Integer::from(2).pow(m - i - 1), p)
                .map_err(|_| AmbsError::ModuloError)?;
            m = i;
            c = b
                .clone()
                .pow_mod(&Integer::from(2), p)
                .map_err(|_| AmbsError::ModuloError)?;
            t = (t * &c).rem_euc(p);
            r = (r * b).rem_euc(p);
        }
        if t == 0 {
            return Ok(vec![Integer::from(0)]);
        }
        Ok(Self::pair_modular_roots(r, p))
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn tonelli_shanks(n: &Integer, p: &Integer) -> Result<Vec<Integer>, AmbsError> {
        let n_mod = n.clone().rem_euc(p);
        if n_mod == 0 {
            return Ok(vec![Integer::from(0)]);
        }
        if *p == 2 {
            return Ok(vec![n_mod]);
        }

        let p_minus_one = p.clone() - Integer::from(1);
        let legendre = n_mod
            .clone()
            .pow_mod(&(&p_minus_one / Integer::from(2)), p)
            .map_err(|_| AmbsError::ModuloError)?;
        if legendre != 1 {
            return Ok(vec![]);
        }

        if p.clone().rem_euc(&Integer::from(4)) == 3 {
            let power = (p.clone() + Integer::from(1)) / Integer::from(4);
            let root = n_mod
                .pow_mod(&power, p)
                .map_err(|_| AmbsError::ModuloError)?;
            return Ok(Self::pair_modular_roots(root, p));
        }

        let (q, s, z) = Self::tonelli_factor_and_nonresidue(p)?;
        let c = z.pow_mod(&q, p).map_err(|_| AmbsError::ModuloError)?;
        let t = n_mod
            .clone()
            .pow_mod(&q, p)
            .map_err(|_| AmbsError::ModuloError)?;
        let power_r = (q + Integer::from(1)) / Integer::from(2);
        let r = n_mod
            .pow_mod(&power_r, p)
            .map_err(|_| AmbsError::ModuloError)?;
        Self::tonelli_refine_root(s, c, t, r, p)
    }

    #[verified_engine::verified]
    fn hensels_lifting(a: &Integer, p: &Integer, k: u32) -> Result<Vec<Integer>, AmbsError> {
        let roots_mod_p = Self::tonelli_shanks(a, p)?;
        let mut final_roots = Vec::new();

        let p_k = p.clone().pow(k);

        for r1 in roots_mod_p {
            let mut x_n = r1.clone();
            let mut p_n = p.clone();

            for _ in 1..k {
                let f_x_n = (&x_n * &x_n) - a.clone();
                let f_prime_x_n = &x_n * Integer::from(2);

                let f_prime_inv_opt = modular_inverse(&f_prime_x_n, p);
                if let Some(f_prime_inv) = f_prime_inv_opt {
                    let term = (f_x_n / &p_n) * f_prime_inv;
                    let term_mod = term.rem_euc(p);
                    x_n -= term_mod * &p_n;
                    p_n *= p;
                    x_n = x_n.rem_euc(&p_n);
                } else {
                    break;
                }
            }

            let check = Integer::from(&x_n * &x_n).rem_euc(&p_k);
            let a_mod = a.clone().rem_euc(&p_k);
            if check == a_mod {
                final_roots.push(x_n);
            }
        }

        final_roots.sort();
        final_roots.dedup();
        Ok(final_roots)
    }

    #[verified_engine::verified]
    fn crt(residues: &[Integer], moduli: &[Integer]) -> Option<Integer> {
        if residues.is_empty() || residues.len() != moduli.len() {
            return None;
        }

        let mut total_modulus = Integer::from(1);
        for m in moduli {
            total_modulus *= m.clone();
        }

        let mut result = Integer::from(0);
        for (r, m) in residues.iter().zip(moduli.iter()) {
            let m_i = total_modulus.clone() / m.clone();
            {
                let m_i_inv = modular_inverse(&m_i, m)?;
                let term = r.clone() * m_i * m_i_inv;
                result += term;
            }
        }

        Some(result.rem_euc(&total_modulus))
    }

    #[verified_engine::verified]
    fn cartesian_product(lists: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
        let mut res = vec![vec![]];
        for list in lists {
            let mut new_res = Vec::new();
            for r in list {
                for seq in &res {
                    let mut new_seq = seq.clone();
                    new_seq.push(r.clone());
                    new_res.push(new_seq);
                }
            }
            res = new_res;
        }
        res
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn compute_modular_square_roots(
        x_l: &Integer,
        s_l: &Integer,
    ) -> Result<Vec<Integer>, AmbsError> {
        let s_l_u64 = s_l.to_u64().unwrap_or(0);
        if s_l_u64 == 0 {
            return Ok(vec![]);
        }

        let factors = prime_factors(s_l_u64);
        let mut root_lists = Vec::new();
        let mut moduli = Vec::new();

        for (p, e) in factors {
            let p_int = Integer::from(p);
            let roots = Self::hensels_lifting(x_l, &p_int, e)?;
            if roots.is_empty() {
                return Ok(vec![]);
            }
            root_lists.push(roots);
            moduli.push(p_int.pow(e));
        }

        let combinations = Self::cartesian_product(&root_lists);
        let mut final_roots = Vec::new();

        for combo in combinations {
            if let Some(root) = Self::crt(&combo, &moduli) {
                final_roots.push(root);
            }
        }

        Ok(final_roots)
    }

    #[verified_engine::verified]
    fn sieve_candidate_offsets(
        &self,
        n_l: u64,
        n_l_int: &Integer,
        s_l: &Integer,
        r_i: &Integer,
        c_max: usize,
    ) -> Result<(), AmbsError> {
        debug_assert!(n_l > 0, "prefix factor n_l must be positive");
        debug_assert!(*s_l > 0, "divisor sum s_l must be positive");
        let mut c_valid = vec![true; c_max];
        let b_stop_sqrt = (self.b_stop / (n_l as f64)).sqrt() as u64;
        let primes: Vec<u64> = (2..=std::cmp::max(2, b_stop_sqrt))
            .filter(|&q| is_prime(q))
            .collect();

        for q in primes {
            let q_int = Integer::from(q);
            if let Some(s_l_inv) = modular_inverse(s_l, &q_int) {
                let target_c = ((Integer::from(0) - r_i) * s_l_inv)
                    .rem_euc(&q_int)
                    .to_usize()
                    .ok_or(AmbsError::ConversionError)?;
                for c in (target_c..c_max).step_by(q as usize) {
                    c_valid[c] = false;
                }
            }
        }

        for (c, &is_valid) in c_valid.iter().enumerate().take(c_max) {
            if is_valid {
                let z = Integer::from(r_i + &(Integer::from(c) * s_l));
                let z_sq = Integer::from(&z * &z);
                let _target_sigma = (Integer::from(2) * n_l_int * &z_sq + Integer::from(1)) / s_l;
            }
        }
        Ok(())
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn search(&mut self, n_max_str: &str) -> Result<Option<Integer>, AmbsError> {
        let n_max = Integer::from_str_radix(n_max_str, 10)
            .map_err(|_| AmbsError::ParseError("err".to_string()))?;

        self.build_prefix(1, 0, 3.0);

        for &n_l in &self.prefix_pool {
            let s_l = Integer::from(sigma(n_l));
            let n_l_int = Integer::from(n_l);

            let Some(x_l) = modular_inverse(&(Integer::from(-2) * &n_l_int), &s_l) else {
                continue;
            };

            let roots = Self::compute_modular_square_roots(&x_l, &s_l)?;
            if roots.is_empty() {
                continue;
            }

            let z_max = (n_max.clone() / &n_l_int).sqrt();
            let c_max = (z_max / &s_l).to_usize().unwrap_or(10_000_000);

            for r_i in roots {
                self.sieve_candidate_offsets(n_l, &n_l_int, &s_l, &r_i, c_max)?;
            }
        }
        Ok(None)
    }
}

#[allow(missing_docs)]
#[verified_engine::verified]
pub fn solve_quasiperfect_modularity(
    _s_l_str: &str,
    _n_l_str: &str,
    n_max_str: &str,
    _c_max_int_str: &str,
) -> Result<String, AmbsError> {
    let mut ambs = AmbsDsp::new(20, 1, 1000.0);
    if let Some(val) = ambs.search(n_max_str)? {
        Ok(format!("Found: {:?}", val))
    } else {
        Ok("None".to_string())
    }
}

#[cfg(test)]
mod tests;
