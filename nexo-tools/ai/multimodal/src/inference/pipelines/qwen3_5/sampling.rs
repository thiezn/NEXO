use local_inference_helpers::candle_core::{Tensor, D};

pub fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> anyhow::Result<u32> {
    if temperature == 0.0 {
        sample_greedy(logits)
    } else {
        sample_top_p(logits, temperature, top_p)
    }
}

fn sample_greedy(logits: &Tensor) -> anyhow::Result<u32> {
    Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?)
}

fn sample_top_p(logits: &Tensor, temperature: f64, top_p: f64) -> anyhow::Result<u32> {
    use rand::prelude::Distribution;

    let scaled = (logits / temperature)?;
    let probs =
        local_inference_helpers::candle_nn::ops::softmax_last_dim(&scaled.unsqueeze(0)?)?
            .squeeze(0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

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
