# burnoutIA/llama-cpp-rdna-boosts — the `harn` branch

A fork of [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) carrying two extra
blocks on top of the upstream release `v16-4f9dfa0e3-r289`. It is the engine that
[Harn](https://github.com/burnoutIA) builds for RDNA4 cards. Everything else (base, blocks `0000`–`0015`, `0300`,
the scripts and the docs) is upstream, unchanged.

| block | what | scope |
|---|---|---|
| `0090` | speculative verify (2–16 columns) in WMMA iu8 | RDNA4 only (gfx1200/gfx1201) |
| `0091` | GDN conv + q/k norm fusion no longer refused on a deferred conv-state gather | every GPU the HIP/CUDA path serves |

`release.json` records the new default tree (`v18-4f9dfa0e3-r289-harn1`, tree `01460a79b0f4…`); `scripts/apply-all.sh`
applies the 19 default patches strictly, as upstream.

## 0090 — speculative verify in WMMA iu8 (RDNA4)

The ksplit `mul_mat_vec_q` path recomputes dequantization and int8 dot products for every verified token. At
7–8 columns (a DFlash2 or deep-MTP verify) it becomes instruction-bound: on a Radeon AI PRO R9700 the verify
matvecs of Qwen3.8-27B UD-Q4_K_M cost 34.3 ms at 8 tokens against 25.9 ms at 1.

`0090` computes the same products with `wmma_i32_16x16x16_iu8` for 2–16 columns, dequantizing each weight once
for all columns (C^T = X^T · W^T: activations are the signed A operand, weights the B operand, so each lane keeps
its weight row and its block scales). Q4_K, Q5_K, Q6_K and IQ4_XS are read in their original GGUF layout (the K
positions inside a 32-block are permuted consistently on both operands), on the existing q8_1 activations. Other
types, shapes and GPUs take the existing path.

- `GGML_CUDA_WMMA_MMVQ=0` disables it; `=n` enables it from n columns (default 2).
- `test-backend-ops -o MUL_MAT` passes; a run-time cross-check against the ksplit path during generation shows a
  relative difference ≈ 2e-7.
- **Not bit-identical to the single-token path** (different accumulation order), so the decode/verify
  greedy-purity guarantee of `GREEDY-PURITY.md` does not hold with it enabled.

R9700, Qwen3.8-27B UD-Q4_K_M, DFlash2 n=7, 256 tokens, temperature 1, 10 seeds:

| engine | code tok/s | prose tok/s | ms per cycle (code) |
|---|---|---|---|
| upstream `v16-4f9dfa0e3-r289` | 124.0 | 51.0 | 47.4 |
| this branch | 130.7 | 55.6 | 42.6 |

## 0091 — GDN conv + q/k norm fusion on a deferred conv-state gather

`gdn_conv_norm_match` refused the conv + q/k l2-norm fusion when the normalized q output overlapped the
conv-state tensor. On the speculative verify path the conv-state gather is deferred: the direct kernel reads
the cache row and never that dead GET_ROWS output, so the overlap is harmless. The q/k norms then ran as a
separate kernel plus one launch in every Gated DeltaNet layer (−0.7 ms per DFlash2 cycle on the 27B once fixed).
Results are bit-identical; `GGML_CUDA_FUSE_GDN_CONV_NORM=0` still disables the fusion.

## Not re-validated

The optional artifacts (`optional/`) and the beta candidates (`beta/`) were cut against the upstream default
tree and have not been re-applied on top of `0090`/`0091`.
