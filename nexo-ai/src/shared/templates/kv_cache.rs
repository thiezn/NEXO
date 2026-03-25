/// Abstraction over KV cache operations for prefix caching.
///
/// Models implement this to allow conversation-aware cache reuse across turns.
/// Phase 1: stub implementations (clear only).
/// Phase 2: real prefix caching with token-level truncation.
pub trait KvCacheState {
    /// Current number of tokens stored in the KV cache.
    fn cache_token_count(&self) -> usize;

    /// Clear the entire KV cache.
    fn clear_cache(&mut self);

    /// Truncate cache to the given length (for prefix caching).
    /// Keeps KV state for tokens [0..len), discards the rest.
    fn truncate_to(&mut self, len: usize);
}
