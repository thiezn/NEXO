use mlx_rs::{Array, error::Exception, ops::concatenate};

pub struct KvCache {
    k: Option<Array>,
    v: Option<Array>,
    offset: i32,
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            offset: 0,
        }
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Append new K, V tensors and return the full accumulated K, V.
    /// K, V shapes: (batch, heads, seq_len, head_dim)
    pub fn append(&mut self, k: &Array, v: &Array) -> Result<(Array, Array), Exception> {
        let (new_k, new_v) = match (self.k.take(), self.v.take()) {
            (Some(prev_k), Some(prev_v)) => {
                let new_k = concatenate(&[&prev_k, k], 2)?;
                let new_v = concatenate(&[&prev_v, v], 2)?;
                (new_k, new_v)
            }
            _ => (k.clone(), v.clone()),
        };

        self.offset = new_k.shape()[2] as i32;
        self.k = Some(new_k.clone());
        self.v = Some(new_v.clone());
        Ok((new_k, new_v))
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.offset = 0;
    }
}
