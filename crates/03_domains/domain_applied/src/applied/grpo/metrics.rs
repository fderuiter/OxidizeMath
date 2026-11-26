use std::collections::HashSet;

#[verified_engine::verified]
fn ngrams(s: &str, n: usize) -> HashSet<String> {
    s.split_whitespace()
        .collect::<Vec<&str>>()
        .windows(n)
        .map(|w| w.join(" "))
        .collect()
}

#[verified_engine::verified]
fn bleu_precision(candidate: &str, reference: &str, n: usize) -> f64 {
    let candidate_ngrams = ngrams(candidate, n);
    let reference_ngrams = ngrams(reference, n);

    if candidate_ngrams.is_empty() {
        return if reference_ngrams.is_empty() {
            1.0
        } else {
            0.0
        };
    }

    let intersection = candidate_ngrams.intersection(&reference_ngrams).count();
    intersection as f64 / candidate_ngrams.len() as f64
}

#[verified_engine::verified]
fn simple_bleu(candidate: &str, reference: &str) -> f64 {
    let p1 = bleu_precision(candidate, reference, 1);
    let p2 = bleu_precision(candidate, reference, 2);
    let p3 = bleu_precision(candidate, reference, 3);
    let p4 = bleu_precision(candidate, reference, 4);

    if p1 == 0.0 {
        return 0.0;
    }

    let candidate_len = candidate.split_whitespace().count();
    if candidate_len == 0 {
        return 0.0;
    }
    let reference_len = reference.split_whitespace().count();

    let brevity_penalty = if candidate_len > reference_len {
        1.0
    } else {
        (1.0 - reference_len as f64 / candidate_len as f64).exp()
    };

    brevity_penalty * (p1 * p2 * p3 * p4).powf(0.25)
}

/// Pairwise distance between questions using a simplified BLEU score.
///
/// # Arguments
///
/// * `question_i` - The first question string.
/// * `question_j` - The second question string.
///
/// # Returns
///
/// The distance between the questions (1.0 - BLEU score).
#[verified_engine::verified]
pub fn pairwise_distance_bleu(question_i: &str, question_j: &str) -> f64 {
    let score = simple_bleu(question_i, question_j);
    1.0 - score
}
