/// XOR-decrypt a buffer in place.
pub fn decrypt(data: &mut [u8], key: u8) {
    for byte in data.iter_mut() {
        *byte ^= key;
    }
}
