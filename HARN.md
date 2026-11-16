# burnoutIA/llama-cpp-rdna-boosts — the `harn` branch

A fork of [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) carrying three extra
blocks on top of the upstream release `v16-4f9dfa0e3-r289`. It is the engine that
[Harn](https://github.com/burnoutIA) builds for RDNA4 cards. Everything else (base, blocks `0000`–`0015`, `0300`,
the scripts and the docs) is upstream, unchanged.

| block | what | scope |
|---|---|---|
| `0090` | speculative verify (2–16 columns) in WMMA iu8 | RDNA4 only (gfx1200/gfx1201) |
| `0091` | GDN conv + q/k norm fusion no longer refused on a deferred conv-state gather | every GPU the HIP/CUDA path serves |
| `0092` | MTP draft head: coarse low-bit copy over the full vocabulary, exact rerank (opt-in) | Qwen3.5/3.6/3.8 MTP, any backend |

`release.json` records the new default tree (`v19-4f9dfa0e3-r289-harn2`, tree `345e62faed51…`); `scripts/apply-all.sh`
applies the 20 default patches strictly, as upstream.

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

## 0092 — coarse MTP draft head with exact rerank (opt-in)

Every MTP draft step reads the LM head (248k × 5120, Q6_K: 1.04 GB on Qwen3.8-27B). `--spec-draft-mtp-vocab N`
(block `0013`) cuts that read by scoring only the first N token ids, which costs acceptance on prose. With
`LLAMA_MTP_DRAFT_COARSE=K` set as well, the drafter keeps the full vocabulary: it scores every token with a
low-bit copy of the head (`LLAMA_MTP_DRAFT_COARSE_TYPE`, default `q2_K`; `q2_0` measured best), requantized once
on the CPU at load (+4.5 s on the 27B), then rescores the best K exactly against the original head. Only the draft
proposal changes; the target verifies with its own head. The idea comes from vllm-radiance's 2-bit draft head.

R9700, Qwen3.8-27B Uncensored Q4_K_M, 10 seeds, 256 tokens, temperature 1 (code / prose tok/s, prose acceptance):

| draft head | code | prose | prose acceptance |
|---|---|---|---|
| MTP3, vocab 98304 | 88.8 | 48.2 | 0.314 |
| MTP3, full vocab | 82.7 | 50.0 | 0.386 |
| MTP3, coarse `q2_0`, K=64 | 88.7 | 53.4 | 0.382 |
| adaptive ≤9 + ngram, vocab 98304 | 115.4 | 43.0 | 0.199 |
| adaptive ≤9 + ngram, coarse `q2_0` | 115.1 | 47.4 | 0.232 |

K = 16, 64 and 256 measure the same. The same scheme on the DFlash2 drafter was rejected: its selector reads the
whole distribution, acceptance fell (0.694 → 0.655) for 0.4 ms per cycle.

## Not re-validated

The optional artifacts (`optional/`) and the beta candidates (`beta/`) were cut against the upstream default
tree and have not been re-applied on top of `0090`/`0091`.
