/// LSB-first bit reader matching ScummVM's FILL_BITS/READ_BIT macros.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bits: u32,
    cl: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bits: 0,
            cl: 0,
        }
    }

    /// Create a BitReader with 2 bytes preloaded (16 bits), matching
    /// ScummVM's MajMinCodec::setupBitReader.
    pub fn new_preloaded(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            pos: 0,
            bits: 0,
            cl: 0,
        };
        if data.len() >= 2 {
            reader.bits = data[0] as u32 | ((data[1] as u32) << 8);
            reader.cl = 16;
            reader.pos = 2;
        }
        reader
    }

    fn fill(&mut self) {
        if self.cl <= 24 && self.pos < self.data.len() {
            self.bits |= (self.data[self.pos] as u32) << self.cl;
            self.pos += 1;
            self.cl += 8;
        }
    }

    pub fn read_bit(&mut self) -> u8 {
        self.fill();
        let bit = (self.bits & 1) as u8;
        self.bits >>= 1;
        self.cl = self.cl.saturating_sub(1);
        bit
    }

    pub fn read_bits(&mut self, n: u8) -> u8 {
        self.fill();
        let mask = (1u32 << n) - 1;
        let val = (self.bits & mask) as u8;
        self.bits >>= n;
        self.cl = self.cl.saturating_sub(n);
        val
    }
}
