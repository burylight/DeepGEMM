import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import (
    ceil_div,
    per_token_cast_to_nvfp4,
    cast_back_from_nvfp4,
    unpack_ue4m3_from_int,
)


def unpack_tma_packed_ue4m3(sf: torch.Tensor, sf_k: int) -> torch.Tensor:
    unpacked = unpack_ue4m3_from_int(sf.contiguous())
    return unpacked.view(sf.size(0), sf.size(1) * 4)[:, :sf_k]


def to_tma_packed_ue4m3(sf: torch.Tensor, mn: int, k: int) -> torch.Tensor:
    return deep_gemm.transform_sf_into_required_layout_nvfp4(sf, mn, k)


def run_case(m: int, n: int, k: int, use_packed_sf: bool, use_uint8: bool) -> None:
    a_ref = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b_ref = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

    a_fp4, sfa = per_token_cast_to_nvfp4(a_ref, use_packed_ue4m3=False)
    b_fp4, sfb = per_token_cast_to_nvfp4(b_ref, use_packed_ue4m3=False)
    if use_uint8:
        a_fp4 = a_fp4.to(torch.uint8)
        b_fp4 = b_fp4.to(torch.uint8)

    sfa_packed = to_tma_packed_ue4m3(sfa, m, k)
    sfb_packed = to_tma_packed_ue4m3(sfb, n, k)
    kernel_sfa = sfa_packed if use_packed_sf else sfa
    kernel_sfb = sfb_packed if use_packed_sf else sfb

    ref_sfa = unpack_tma_packed_ue4m3(sfa_packed, ceil_div(k, 16))
    ref_sfb = unpack_tma_packed_ue4m3(sfb_packed, ceil_div(k, 16))
    ref_a = cast_back_from_nvfp4(a_fp4.view(torch.int8), ref_sfa)
    ref_b = cast_back_from_nvfp4(b_fp4.view(torch.int8), ref_sfb)
    ref_d = (ref_a @ ref_b.t()).to(torch.bfloat16)

    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.nvfp4_gemm_nt((a_fp4, kernel_sfa), (b_fp4, kernel_sfb), d)
    diff = calc_diff(d, ref_d)
    assert diff < 0.025, f'{m=}, {n=}, {k=}, {use_packed_sf=}, {use_uint8=}, {diff=:.5f}'
    print(f' > m={m:5}, n={n:5}, k={k:5}, packed_sf={int(use_packed_sf)}, uint8={int(use_uint8)}, diff={diff:.5f}')


def test_nvfp4_gemm() -> None:
    print('Testing NVFP4 GEMM:')
    for m, n, k in ((128, 128, 256), (4096, 7168, 2048), (4096, 7168, 7168)):
        run_case(m, n, k, use_packed_sf=False, use_uint8=False)
        run_case(m, n, k, use_packed_sf=True, use_uint8=True)
    print()


def test_nvfp4_block_n_variants() -> None:
    print('Testing NVFP4 block-N variants:')
    try:
        for block_n in (160, 192, 224):
            deep_gemm.set_block_size_multiple_of((1, block_n))
            run_case(256, block_n * 2, 256, use_packed_sf=True, use_uint8=True)
    finally:
        deep_gemm.set_block_size_multiple_of((1, 1))
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_nvfp4_gemm()
    test_nvfp4_block_n_variants()
