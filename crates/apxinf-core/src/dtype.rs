/// Supported data types for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// NVIDIA/CUDA FP8 E4M3 finite-number encoding.
    F8E4M3,
    /// Raw unsigned byte carrier (used for packed quantized weights and
    /// zero-points before they are reinterpreted by a quantized GEMM).
    U8,
    /// Raw signed byte carrier (int8 zero-points, packed int4 weight bytes).
    I8,
    /// Raw i32 carrier (pack-quantized int4 groups stored 8 nibbles per i32).
    I32,
    /// Raw i64 carrier (weight_shape metadata, 2-element [out, in]).
    I64,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8E4M3 => 1,
            DType::U8 | DType::I8 => 1,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F8E4M3 => write!(f, "f8_e4m3"),
            DType::U8 => write!(f, "u8"),
            DType::I8 => write!(f, "i8"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
        }
    }
}
