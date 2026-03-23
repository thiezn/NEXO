/// Check if data looks like valid MIDI (starts with "MThd").
pub fn is_valid_midi(data: &[u8]) -> bool {
    data.len() >= 4 && &data[0..4] == b"MThd"
}

/// Wrap raw MIDI-like data in a minimal Standard MIDI File if it doesn't already have a header.
/// Many SCUMM ADL/ROL/GMD blocks already contain valid MIDI data, so we just pass through.
/// For data that isn't valid MIDI, we save it as-is (it may be AdLib register dumps).
pub fn ensure_midi_header(data: &[u8]) -> Vec<u8> {
    if is_valid_midi(data) {
        return data.to_vec();
    }
    // Not standard MIDI - return as-is (AdLib register data, etc.)
    data.to_vec()
}
