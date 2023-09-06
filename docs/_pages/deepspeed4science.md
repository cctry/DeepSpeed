---
title: "DeepSpeed4Science Overview and Tutorial"
permalink: /deepspeed4science/
toc: true
toc_label: "Contents"
toc_sticky: true
---

Deep learning is becoming a powerful tool for scientific research. In line with Microsoft's mission to solve humanity's most pressing challenges, the DeepSpeed team at Microsoft is responding to this opportunity by launching a new initiative called *DeepSpeed4Science*, aiming to build unique capabilities through AI system technology innovations to help domain experts to unlock today's biggest science mysteries. Details of the DeepSpeed4Science initiative can be found at [our blog](TODO) and [our website](TODO).

This page serves as the "main tutorial catalogue" for all technologies released (or to be released in the future) as part of the DeepSpeed4Science initiative, making it easier for scientists to shop for techniques they need. For each technique we will introduce what is it for, when to use it, how to use it, and existing scientific applications of the techniques (we welcome users to contribute more showcases if you apply our techniques in your scientific research):

* [2023/09] We are releasing two techniques: [DS4Sci_EvoformerAttention](#1-ds4sci_evoformerattention) and [DeepSpeed4Science large-scale training framework](#2-deepspeed4science-large-scale-training-framework), and their scientific applications in structural biology research.

## 1. DS4Sci_EvoformerAttention

### 1.1 What is DS4Sci_EvoformerAttention
`DS4Sci_EvoformerAttention` is a collection of kernels built to scale the [Evoformer](https://www.nature.com/articles/s41586-021-03819-2) computation to larger number of sequences and residuals by reducing the memory footprint and increasing the training speed.

### 1.2 When to use DS4Sci_EvoformerAttention
`DS4Sci_EvoformerAttention` is most beneficial when the number of sequences and residuals is large. The forward kernel is optimized to accelerate computation. It is beneficial to use the forward kernel during inference for various attention mechanisms. The associated backward kernel can be used during training to reduce the memory footprint at the cost of some computation. Therefore, it is beneficial to use `DS4Sci_EvoformerAttention` in training for memory-constrained operations such as MSA row-wise attention and MSA column-wise attention.

### 1.3 How to use DS4Sci_EvoformerAttention

**1.3.1 Installation**

`DS4Sci_EvoformerAttention` is released as part of DeepSpeed >= 0.10.3. `DS4Sci_EvoformerAttention` is implemented based on [CUTLASS](https://github.com/NVIDIA/cutlass). You need to clone the CUTLASS repository and specify the path to it in the environment variable `CUTLASS_PATH`.

```shell
git clone https://github.com/NVIDIA/cutlass
export CUTLASS_PATH=/path/to/cutlass
```
The kernels will be compiled when `DS4Sci_EvoformerAttention` is called for the first time.

`DS4Sci_EvoformerAttention` requires GPUs with compute capability 7.0 or higher (NVIDIA V100 or later GPUs) and the minimal CUDA version is 11.3. It is recommended to use CUDA 11.7 or later for better performance. Besides, the performance of backward kernel on V100 kernel is not as good as that on A100 for now.

**1.3.2 Unit test and benchmark**

The unit test and benchmark are available in the `tests` folder in DeepSpeed repo. You can use the following command to run the unit test and benchmark.

```shell
pytest -s tests/unit/ops/deepspeed4science/test_DS4Sci_EvoformerAttention.py
python tests/benchmarks/DS4Sci_EvoformerAttention_bench.py
```

**1.3.3 Applying DS4Sci_EvoformerAttention to your own model**

To use `DS4Sci_EvoformerAttention` in user's own models, you need to import `DS4Sci_EvoformerAttention` from `deepspeed.ops.deepspeed4science`.

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

### 1.4 DS4Sci_EvoformerAttention scientific application

**1.4.1 DS4Sci_EvoformerAttention eliminates memory explosion problems for scaling Evoformer-centric structural biology models in OpenFold**

[OpenFold](https://github.com/aqlaboratory/openfold) is a community reproduction of DeepMind's AlphaFold2 that makes it possible to train or finetune AlphaFold2 on new datasets. Training AlphaFold2 incurs a memory explosion problem because it contains several custom Evoformer attention variants that manifest unusually large activations. By leveraging DeepSpeed4Science's DS4Sci_EvoformerAttention kernels, OpenFold team is able to reduce the peak memory requirement by 13x without accuracy loss. Detailed information about this application can be found at [our blog](TODO) and [our website](TODO). OpenFold team also hosts an [example](TODO) about how to use DS4Sci_EvoformerAttention in the OpenFold repo.

## 2. DeepSpeed4Science large-scale training framework

### 2.1 What is DeepSpeed4Science large-scale training framework
DeepSpeed4Science large-scale training framework is a set of systematic and algorithmic approaches for enabling larger-scale model training, especially when very long sequences is used. Specifically, DeepSpeed4Science large-scale training framework is released as an update of the [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) framework. This update includes a rebase with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) to leverage NVIDIA's newest technologies, and several DeepSpeed's own optimizations on attention map memory optimization and position embedding partitioning.

### 2.2 When to use DeepSpeed4Science large-scale training framework
The updated Megatron-DeepSpeed framework can be used for any large-scale model training. It is particularly beneficial to use our framework when you have out-of-memory issues when training with long sequence lengths. Our results show that our long sequence support enables up to 9x longer sequence lengths than NVIDIA's Megatron-LM.

In parallel to the Megatron-LM-based long sequence support that we extend and improve, [DeepSpeed-Ulysses](/tutorials/ds-sequence/) is another DeepSpeed technology aiming to support long sequences by reducing the communication via quantization. DeepSpeed-Ulysses is also integrated in the updated Megatron-DeepSpeed framework. The two long sequence support approaches [cannot be used at the same time](https://github.com/microsoft/DeepSpeed/issues/4217). We recommend users to try both of them.

### 2.3 How to use DeepSpeed4Science large-scale training framework
We have a [detailed tutorial](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed/deepspeed4science/megatron_long_seq_support) in the Megatron-DeepSpeed repo about how to use the framework when having long sequences, including installation/setup instructions, example scripts, and reference results.

### 2.4 DeepSpeed4Science large-scale training framework scientific application

**2.4.1 DeepSpeed4Science long sequence support enables very-long sequence support for genome-scale foundation models in GenSLMs**

[GenSLMs](https://github.com/ramanathanlab/genslm), a 2022 [ACM Gordon Bell award](https://www.acm.org/media-center/2022/november/gordon-bell-special-prize-covid-research-2022) winning genome-scale language model from Argonne National Lab, can learn the evolutionary landscape of SARS-CoV-2 (COVID-19) genomes by adapting large language models (LLM’s) for genomic data. To achieve their scientific goal, GenSLMs and similar models require very long sequence support for both training and inference that is beyond generic LLM's long-sequence strategies. By leveraging DeepSpeed4Science's long sequence support, GenSLMs team is able to train their 25B model with 512K sequence length, much longer than their original 42K sequence length. Detailed information about this application can be found at [our blog](TODO) and [our website](TODO). GenSLMs team also hosts an [example](TODO) about how to use DeepSpeed4Science long sequence support in the GenSLMs repo.
