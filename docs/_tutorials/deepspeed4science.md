---
title: "DS4Sci_EvoformerAttention"
tags: training
---

**What is DS4Sci_EvoformerAttention:** `DS4Sci_EvoformerAttention` is a collection of kernels built to scale the Evoformer computation to larger number of sequences and residuals by reducing the memory footprint and increasing the training speed.

**When to use DS4Sci_EvoformerAttention**: `DS4Sci_EvoformerAttention` is most beneficial when the number of sequences and residuals is large. The forward kernel is optimized to accelerate computation. It is beneficial to use the forward kernel during inference for various attention mechanisms. The associated backward kernel can be used during training to reduce the memory footprint at the cost of some computation. Therefore, it is beneficial to use `DS4Sci_EvoformerAttention` in training for memory-constrained operations such as MSA row-wise attention and MSA column-wise attention.

**How to use DS4Sci_EvoformerAttention**:

(1) `DS4Sci_EvoformerAttention` is implemented based on [CUTLASS](https://github.com/NVIDIA/cutlass). We need to clone the CUTLASS repository and specify the path to CUTLASS in the environment variable `CUTLASS_PATH`.

```shell
git clone https://github.com/NVIDIA/cutlass
export CUTLASS_PATH=/path/to/cutlass
```
The kernels will be compiled when `DS4Sci_EvoformerAttention` is called for the first time.

`DS4Sci_EvoformerAttention` requires GPUs with compute capability 7.0 or higher (V100 or later GPUs) and the minimal CUDA version is 11.3. It is recommended to use CUDA 11.7 or later for better performance. Besides, the performance of backward kernel on V100 kernel is not as good as that on A100 for now.

(2) The unit test and benchmark are available in the `tests`. We can use the following command to run the unit test and benchmark.

```shell
pytest -s tests/unit/ops/deepspeed4science/test_DS4Sci_EvoformerAttention.py
python tests/benchmarks/DS4Sci_EvoformerAttention_bench.py
```

(3) To use `DS4Sci_EvoformerAttention` in user's own models, we need to import `DS4Sci_EvoformerAttention` from `deepspeed.ops.deepspeed4science`.

```python
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
```

`DS4Sci_EvoformerAttention` supports four attention mechanisms in Evoformer by using different inputs as shown in the following examples. In the examples, we denote the number of sequences as `N_seq` and the number of residuals as `N_res`. The dimension of the hidden states `Dim` and head number `Head` are different among different attention. Note that `DS4Sci_EvoformerAttention` requires the input tensors to be in `torch.float16` or 'torch.bfloat16' data type.

(a) **MSA row-wise attention** builds attention weights for residue pairs and integrates the information from the pair representation as an additional bias term.
```python
# Q, K, V: [Batch, N_seq, N_res, Head, Dim]
# res_mask: [Batch, N_seq, 1, 1, N_res]
# pair_bias: [Batch, 1, Head, N_res, N_res]
out = DS4Sci_EvoformerAttention(Q, K, V, [res_mask, pair_bias])
```

(b) **MSA column-wise attention** lets the elements that belong to the same target residue exchange information.
```python
# Q, K, V: [Batch, N_res, N_seq, Head, Dim]
# res_mask: [Batch, N_seq, 1, 1, N_res]
out = DS4Sci_EvoformerAttention(Q, K, V, [res_mask])
```

(b) **Triangular self-attention** updates the pair representation. Below is the example of triangular self-attention around starting node. The triangular self-attention around ending node is similar.
```python
# Q, K, V: [Batch, N_res, N_res, Head, Dim]
# res_mask: [Batch, N_res, 1, 1, N_res]
# right_edges: [Batch, 1, Head, N_res, N_res]
out = DS4Sci_EvoformerAttention(Q, K, V, [res_mask, right_edges])
```
