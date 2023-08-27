---
title: "DeepSpeed4Science: Enabling large-scale scientific discovery through sophisticated AI system technologies"
tags: training inference
---

The DeepSpeed4Science initiative aims to build AI system technology innovations to help domain experts to tackle science challenges using deep learning. Details of the initiative can be found in our [blog](TODO) and our [website](TODO). This page includes tutorials of all technologies released as part of the DeepSpeed4Science initiative.

## 1. DeepSpeed4Science Evoformer Attention (DS4Sci_EvoformerAttention)

### 1.1 What is DS4Sci_EvoformerAttention
`DS4Sci_EvoformerAttention` is a collection of kernels built to scale the Evoformer computation to larger number of sequences and residuals by reducing the memory footprint and increasing the training speed.

### 1.2 When to use DS4Sci_EvoformerAttention
`DS4Sci_EvoformerAttention` is most beneficial when the number of sequences and residuals is large. The forward kernel is optimized to accelerate computation. It is beneficial to use the forward kernel during inference for various attention mechanisms. The associated backward kernel can be used during training to reduce the memory footprint at the cost of some computation. Therefore, it is beneficial to use `DS4Sci_EvoformerAttention` in training for memory-constrained operations such as MSA row-wise attention and MSA column-wise attention.

### 1.3 How to use DS4Sci_EvoformerAttention

#### 1.3.1 Installation
`DS4Sci_EvoformerAttention` is implemented based on [CUTLASS](https://github.com/NVIDIA/cutlass). We need to clone the CUTLASS repository and specify the path to CUTLASS in the environment variable `CUTLASS_PATH`.

```shell
git clone https://github.com/NVIDIA/cutlass
export CUTLASS_PATH=/path/to/cutlass
```
The kernels will be compiled when `DS4Sci_EvoformerAttention` is called for the first time.

`DS4Sci_EvoformerAttention` requires GPUs with compute capability 7.0 or higher (NVIDIA V100 or later GPUs) and the minimal CUDA version is 11.3. It is recommended to use CUDA 11.7 or later for better performance. Besides, the performance of backward kernel on V100 kernel is not as good as that on A100 for now.

#### 1.3.2 Unit test and benchmark
The unit test and benchmark are available in the `tests`. We can use the following command to run the unit test and benchmark.

```shell
pytest -s tests/unit/ops/deepspeed4science/test_DS4Sci_EvoformerAttention.py
python tests/benchmarks/DS4Sci_EvoformerAttention_bench.py
```

#### 1.3.3 Applying DS4Sci_EvoformerAttention to your own model
To use `DS4Sci_EvoformerAttention` in user's own models, we need to import `DS4Sci_EvoformerAttention` from `deepspeed.ops.deepspeed4science`.

```python
from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
```

`DS4Sci_EvoformerAttention` supports four attention mechanisms in Evoformer (MSA row-wise, MSA column-wise, and 2 kinds of Triangular) by using different inputs as shown in the following examples. In the examples, we denote the number of sequences as `N_seq` and the number of residuals as `N_res`. The dimension of the hidden states `Dim` and head number `Head` are different among different attention. Note that `DS4Sci_EvoformerAttention` requires the input tensors to be in `torch.float16` or `torch.bfloat16` data type.

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

(c) **Triangular self-attention** updates the pair representation. There are two kinds of Triangular self-attention: around starting and around ending node. Below is the example of triangular self-attention around starting node. The triangular self-attention around ending node is similar.
```python
# Q, K, V: [Batch, N_res, N_res, Head, Dim]
# res_mask: [Batch, N_res, 1, 1, N_res]
# right_edges: [Batch, 1, Head, N_res, N_res]
out = DS4Sci_EvoformerAttention(Q, K, V, [res_mask, right_edges])
```

#### 1.3.4 Real-world application: DeepSpeed4Science eliminates memory explosion problems for scaling Evoformer-centric structural biology models via DS4Sci_EvoformerAttention
TODO for Gustaf: Need a brief description of the real-world DS4Sci_EvoformerAttention application inside OpenFold and a link to an actual example script/code.

## 2. DeepSpeed4Science long sequence support via both systematic and algorithmic approaches

### 2.1 What is long sequence support
TODO for Chengming/Minjia: Need a brief description about what is the technology.

### 2.2 When to use long sequence support
TODO for Chengming/Minjia: Need a brief description about when to use the technology.

### 2.3 How to use long sequence support
TODO for Chengming/Minjia: Need a brief description about how to use the technology.

TODO for GenSLMs team: Need a brief description of the real-world DeepSpeed4Science long sequence support application inside GenSLMs and a link to an actual example script/code.
