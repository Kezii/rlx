# Ternary (BitNet b1.58) models → FPGA

Take an f32 model, make its weights ternary `{-1, 0, 1}`, export
SystemVerilog, build a Xilinx bitstream with openXC7, and run it on the
board over UART.

```text
f32 Graph ──► QAT (inject_bitnet_qat) ──► train ──► ExportQuantMode::Ternary
                                                     │
          rlx_fpga::reference (bit-exact) ◄──────────┤
                                                     ▼
              SystemVerilog + .mem ──► synth.sh (openXC7) ──► out.bit
                                                     │
                     program.sh (openFPGALoader) ──► board ◄── run_uart.py
```

Runnable end to end:

```sh
cargo run -p rlx-runtime --release --example bitnet_ternary_fpga --features cpu,fpga -- out xc7:arty-a7-35
cd out && ./synth.sh && ./program.sh && python3 run_uart.py tb_image.mem
```

On its synthetic 4-class task the example prints (test accuracy):

| Weights | Accuracy |
|---------|----------|
| f32 | 81.8 % |
| ternary, PTQ only | 55.9 % |
| ternary, BitNet QAT | 82.2 % |
| FPGA integer reference (QAT weights, static int8 activations) | 82.4 % |

PTQ alone loses ~26 points; QAT recovers f32 accuracy.

## 1. The ternary definition — `rlx_ir::ternary`

One definition, shared by training and export so they cannot drift:

```text
weights      γ  = max(mean|W|, ε)                 absmean, per tensor (BitNet)
             Wq = clamp(round(W / γ), -1, 1)       round = half-to-even
activations  s  = 127 / max(max|x|, ε)            absmax, per token
             xq = clamp(round(x · s), -128, 127)
```

`ternarize(w, rows, cols, &TernaryConfig::default())` returns the codes,
`γ`, `dequantize()` and `sparsity()` (the share of MACs the hardware skips).
`TernaryScale::PerChannel` gives one `γ` per output row instead — free on
the FPGA (requant is per channel anyway) but **must match what training
used**.

## 2. Post-training (PTQ)

No training needed — export the f32 graph with
`ExportQuantMode::Ternary`. Works for models already trained ternary
(BitNet checkpoints); for ordinary f32 models expect a large accuracy drop
(table above). Use QAT.

## 3. Quantization-aware training — `rlx_compile::inject_bitnet_qat`

A graph rewrite; no new op, so it trains on every backend that runs
`Abs` / `Reduce` / `Round` / `Clamp` / `StopGradient`:

```rust,ignore
use rlx_compile::{BitNetQatConfig, inject_bitnet_qat};

let cfg = BitNetQatConfig::default()             // per-tensor γ, int8 per-token activations
    .keep_full_precision(["embed", "lm_head"]);  // BitNet keeps these f32
let (qat_graph, report) = inject_bitnet_qat(&train_graph, &cfg);
// report.ternary_weights: leaves now trained ternary; report.skipped: why others were not
let bwd = rlx_autodiff::grad_with_loss(&qat_graph, &weight_ids);
```

Every `MatMul` / `Conv` whose weight operand traces to a leaf (`Param` /
`Constant` / `Input`, through `Transpose` / `Reshape` / `Cast` / `Expand`)
gets

```text
w_q = w + stop_gradient(γ·clamp(round(w/γ), −1, 1) − w)
x_q = x + stop_gradient(int8_tok(x) − x)            (MatMul activations)
```

— the BitNet straight-through estimator: forward sees the quantized values,
backward is exactly identity to the f32 master weights.
`tests/bitnet_qat.rs` checks forward loss and `dW` on CPU against the host
definition. Practical notes: fine-tune from the f32 weights rather than
training from scratch on small models, with a lower learning rate (the
example uses Adam 1e-2 for f32, 2e-3 for QAT).

## 4. Export — `ExportQuantMode::Ternary`

```rust,ignore
ExportSession::fpga("out")
    .quant_mode(rlx_fpga::ExportQuantMode::Ternary)
    .hw_target(rlx_fpga::HwTarget::xc7_board(rlx_fpga::xc7::Xc7Board::ArtyA7_35))
    .legalize(opts)   // weights_f32, act_scales (calibrated), ternary: TernaryConfig
    .export(&inference_graph)?;
```

Weights are ternarized with the same `TernaryConfig` and packed 2 bits per
weight; the multiplier-free add/sub/skip MAC is used on every tune preset.
The FPGA datapath requantizes activations with **static per-tensor** int8
scales (`LegalizeOptions::act_scales`, calibrate them on training data),
not the per-token scales QAT uses — the example shows the gap is small.

CLI / Python: `--hw xc7:arty-a7-35`, `pyrlx.export_fpga(..., quant="ternary", hw="xc7:basys3")`.

## 5. Bitstream and run (openXC7)

See [fpga-export.md § openXC7 boards](fpga-export.md#openxc7-boards-hwtargetxc7_board---hw-xc7board)
for the generated files, toolchain variables, and the UART protocol.
Supported presets: Arty A7-35/100, Basys 3, Nexys A7-100.

## What is verified where

| Check | Where |
|-------|-------|
| Ternary codes / scales / per-token activations | `rlx-ir` `ternary` unit tests |
| QAT forward + STE gradient on CPU | `rlx-runtime/tests/bitnet_qat.rs` |
| f32 graph → ternary FPGA model (codes, accuracy, DSP count) | `rlx-fpga/tests/ternary.rs` |
| Emitted RTL simulated vs reference (2/4/8-bit × tunes, MNIST) | `rlx-fpga/tests/sim_parity.rs` |
| UART board shell, RTL + `synth_xilinx` netlist vs reference | `rlx-fpga/tests/xc7_uart.rs` |
| Full openXC7 flow → `out.bit`, post-route fmax ≥ board clock (opt-in, Docker) | `rlx-fpga/tests/openxc7_docker.rs` |

Not verified in-tree: on-board runs (needs a board). The example's Arty A7-35
build routes at ~165 MHz against the 100 MHz board clock.

## Scope

The per-layer FPGA path covers feed-forward Conv / Dense / ReLU / MaxPool /
Argmax chains (conv with any stride and zero padding). BitNet **transformers** (attention,
RMSNorm, softmax, KV state) are not yet exportable; QAT itself works on
any graph.
## License

MIT OR Apache-2.0.
