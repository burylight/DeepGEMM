import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import ceil_div, per_token_cast_to_fp4, cast_back_from_fp4


def unpack_tma_packed_ue8m0(sf: torch.Tensor, sf_k: int) -> torch.Tensor:
    sf = sf.contiguous()
    unpacked = (sf.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    return unpacked.view(sf.size(0), sf.size(1) * 4)[:, :sf_k]


def to_tma_packed_ue8m0(sf: torch.Tensor, mn: int, k: int) -> torch.Tensor:
    return deep_gemm.transform_sf_into_required_layout(
        sf, mn, k, (1, 32), None, None, False)


def run_case(m: int, n: int, k: int, use_packed_sf: bool, use_uint8: bool) -> None:
    a_ref = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b_ref = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

    a_fp4, sfa = per_token_cast_to_fp4(a_ref, use_ue8m0=False, gran_k=32)
    b_fp4, sfb = per_token_cast_to_fp4(b_ref, use_ue8m0=False, gran_k=32)
    if use_uint8:
        a_fp4 = a_fp4.to(torch.uint8)
        b_fp4 = b_fp4.to(torch.uint8)

    sfa_packed = to_tma_packed_ue8m0(sfa, m, k)
    sfb_packed = to_tma_packed_ue8m0(sfb, n, k)
    kernel_sfa = sfa_packed if use_packed_sf else sfa
    kernel_sfb = sfb_packed if use_packed_sf else sfb

    ref_sfa = unpack_tma_packed_ue8m0(sfa_packed, ceil_div(k, 32))
    ref_sfb = unpack_tma_packed_ue8m0(sfb_packed, ceil_div(k, 32))
    ref_a = cast_back_from_fp4(a_fp4.view(torch.int8), ref_sfa, gran_k=32)
    ref_b = cast_back_from_fp4(b_fp4.view(torch.int8), ref_sfb, gran_k=32)
    ref_d = (ref_a @ ref_b.t()).to(torch.bfloat16)

    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.mxfp4_gemm_nt((a_fp4, kernel_sfa), (b_fp4, kernel_sfb), d)
    diff = calc_diff(d, ref_d)
    assert diff < 0.02, f'{m=}, {n=}, {k=}, {use_packed_sf=}, {use_uint8=}, {diff=:.5f}'
    print(f' > m={m:5}, n={n:5}, k={k:5}, packed_sf={int(use_packed_sf)}, uint8={int(use_uint8)}, diff={diff:.5f}')


def test_mxfp4_gemm() -> None:
    print('Testing MXFP4 GEMM:')
    for m, n, k in ((128, 128, 256), (4096, 7168, 2048), (4096, 7168, 7168)):
        run_case(m, n, k, use_packed_sf=False, use_uint8=False)
        run_case(m, n, k, use_packed_sf=True, use_uint8=True)
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_mxfp4_gemm()
