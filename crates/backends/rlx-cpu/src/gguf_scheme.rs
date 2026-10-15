// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Map [`rlx_gguf::GgmlType`] to RLX [`QuantScheme`] for graph builders.

use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;

/// Returns the RLX [`QuantScheme`] for a GGUF tensor dtype when the type
/// is a supported packed-quant layout for `Op::DequantMatMul`.
pub fn quant_scheme_for_ggml(dtype: GgmlType) -> Option<QuantScheme> {
    rlx_gguf::quant_scheme(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_scheme_for_ggml_legacy_blocks() {
        assert_eq!(
            quant_scheme_for_ggml(GgmlType::Q4_0),
            Some(QuantScheme::GgufQ4_0)
        );
        assert_eq!(
            quant_scheme_for_ggml(GgmlType::Q5_1),
            Some(QuantScheme::GgufQ5_1)
        );
        assert_eq!(quant_scheme_for_ggml(GgmlType::F32), None);
    }
}
