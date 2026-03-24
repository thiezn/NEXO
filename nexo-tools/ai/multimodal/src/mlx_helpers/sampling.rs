#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use mlx_rs::{Array, array, ops, transforms};

pub fn sample_token(logits: &Array, temperature: f64, top_p: f64) -> anyhow::Result<u32> {
    if temperature == 0.0 {
        sample_greedy(logits)
    } else {
        sample_top_p(logits, temperature, top_p)
    }
}

fn sample_greedy(logits: &Array) -> anyhow::Result<u32> {
    let token = ops::argmax(logits, -1, false)?;
    transforms::eval(&[&token])?;
    Ok(token.item::<u32>())
}

fn sample_top_p(logits: &Array, temperature: f64, top_p: f64) -> anyhow::Result<u32> {
    use rand::prelude::Distribution;

    let scaled = logits.multiply(array!(1.0 / temperature as f32))?;
    let probs = ops::softmax(&scaled, &[-1])?;
    transforms::eval(&[&probs])?;

    let probs_vec: Vec<f32> = probs.as_slice().to_vec();

    let mut sorted: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut filtered = Vec::new();
    for (idx, p) in sorted {
        cumsum += p;
        filtered.push((idx, p));
        if cumsum >= top_p as f32 {
            break;
        }
    }

    let total: f32 = filtered.iter().map(|(_, p)| p).sum();
    let weights: Vec<f32> = filtered.iter().map(|(_, p)| p / total).collect();
    let dist = rand::distr::weighted::WeightedIndex::new(&weights)?;
    let chosen = dist.sample(&mut rand::rng());
    Ok(filtered[chosen].0 as u32)
}
