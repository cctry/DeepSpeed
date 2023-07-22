"""
This script is to test the correctness of the EvoformerAttention op.
To run the script,
1. Clone the CUTLASS repo. E.g. git clone https://github.com/NVIDIA/cutlass.git
2. Specify the CUTLASS_PATH environment variable. E.g. export CUTLASS_PATH=$(pwd)/cutlass
3. Run the script. E.g. python run_evoformer_test.py
"""

import torch
from typing import List
from torch.nn import functional as F
from deepspeed.ops.deepspeed4science import Ds4Sci_EvoformerAttention


def attention(q_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
              k_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
              v_input: torch.Tensor,  # [*, Dim_Q, H, C_hid]
              biases: List[torch.Tensor],
              sm_scale: float
              ) -> torch.Tensor:
    # Original shape: [*, Dim_Q, H, C_hid] -> Transpose to: [*, H, Dim_Q, C_hid]
    q = q_input.transpose(-2, -3)
    k = k_input.transpose(-2, -3)
    v = v_input.transpose(-2, -3)

    # Now, q, k, v are in shape: [*, H, Dim_Q, C_hid]
    
    # Transpose k to shape [*, H, C_hid, Dim_Q]
    k_t = k.transpose(-1, -2)

    # Now, q and k_t are in shapes: [*, H, Dim_Q, C_hid] and [*, H, C_hid, Dim_Q] respectively
    
    # [*, H, Dim_Q, Dim_Q]
    a = torch.matmul(q, k_t) * sm_scale

    for b in biases:
        a += b

    a = F.softmax(a, dim=-1)

    # Now, a is in shape [*, H, Dim_Q, Dim_Q], v is in shape [*, H, Dim_Q, C_hid]
    
    # Matmul operation results in [*, H, Dim_Q, C_hid]
    a_v = torch.matmul(a, v)

    # [*, Dim_Q, H, C_hid]
    o = a_v.transpose(-2, -3)

    return o

dtype = torch.float16

batch = 1
N = 256
heads = 4
dim = 32
seq_len = 256


def correctness_test():
    Q = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
    K = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
    V = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=True)
    bias1 = torch.randn(batch, N, 1, 1, seq_len, dtype=dtype, device="cuda", requires_grad=True)
    bias2 = torch.randn(batch, 1, heads, seq_len, seq_len, dtype=dtype, device="cuda", requires_grad=True)
    dout = torch.rand_like(Q, dtype=dtype, device="cuda")
    ref_out = attention(Q, K, V, [bias1, bias2], 1 / (dim ** 0.5))
    ref_out.backward(dout)
    ref_dv, V.grad = V.grad.clone(), None
    ref_dk, K.grad = K.grad.clone(), None
    ref_dq, Q.grad = Q.grad.clone(), None
    ref_db1, bias1.grad = bias1.grad.clone(), None
    ref_db2, bias2.grad = bias2.grad.clone(), None

    out = EvoformerAttention(Q, K, V, [bias1, bias2])
    out.backward(dout)
    dv, v_grad = V.grad.clone(), None
    dk, k_grad = K.grad.clone(), None
    dq, q_grad = Q.grad.clone(), None 
    db1, bias1.grad = bias1.grad.clone(), None
    db2, bias2.grad = bias2.grad.clone(), None

    assert torch.allclose(ref_out, out, atol=2e-2, rtol=0), f"\n{ref_out} \n {out}"
    assert torch.allclose(ref_dv, dv, atol=2e-2, rtol=0), f"\n{ref_dv} \n {dv}"
    assert torch.allclose(ref_dk, dk, atol=2e-2, rtol=0), f"\n{ref_dk} \n {dk}"  
    assert torch.allclose(ref_dq, dq, atol=2e-2, rtol=0), f"\n{ref_dq} \n {dq}"
    assert torch.allclose(ref_db1, db1, atol=2e-2, rtol=1e-2), f"{ref_db1} \n {db1}"
    assert torch.allclose(ref_db2, db2, atol=2e-2, rtol=1e-2), f"{ref_db2} \n {db2}"


def benchmark():
    ours = []
    baseline = []
    for batch_size in range(1, 17):
        Q = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda")
        K = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda")
        V = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda")
        bias1 = torch.randn(batch, N, 1, 1, seq_len, dtype=dtype, device="cuda")
        bias2 = torch.randn(batch, 1, heads, seq_len, seq_len, dtype=dtype, device="cuda")
        # warm up
        EvoformerAttention(Q, K, V, [bias1, bias2])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            EvoformerAttention(Q, K, V, [bias1, bias2])
        end.record()
        torch.cuda.synchronize()
        ours.append(start.elapsed_time(end) / 5)
        # warm up
        attention(Q, K, V, [bias1, bias2], 1 / (dim ** 0.5))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            attention(Q, K, V, [bias1, bias2], 1 / (dim ** 0.5))
        end.record()
        torch.cuda.synchronize()
        baseline.append(start.elapsed_time(end) / 5)
    
    print(f"batch size\tours\tbaseline")
    for i in range(len(ours)):
        print(f"{i+1}\t{ours[i]}\t{baseline[i]}")
        
correctness_test()
benchmark()
