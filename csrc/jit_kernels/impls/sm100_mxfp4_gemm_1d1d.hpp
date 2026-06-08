#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100MXFP4Gemm1D1DRuntime final: public LaunchRuntime<SM100MXFP4Gemm1D1DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        const std::optional<std::string> epilogue_type;

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mxfp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mxfp4_gemm_1d1d_impl<
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {},
        {},
        {}, {},
        {},
        {},
        {}
    >);
}};
)",
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        to_string(args.gemm_desc.cd_dtype),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

static void sm100_mxfp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                  const torch::Tensor& b, const torch::Tensor& sfb,
                                  const torch::Tensor& d,
                                  const int& m, const int& n, const int& k,
                                  const std::string& compiled_dims,
                                  const std::optional<std::string>& epilogue_type = std::nullopt) {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .mma_kind = MmaKind::MXFP4,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM100ArchSpec>(desc);

    const auto tensor_map_a = make_tma_packed_fp4_a_desc(a, m, k,
                                                         config.storage_config.load_block_m,
                                                         config.layout.block_k,
                                                         static_cast<int>(a.stride(0)),
                                                         config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_packed_fp4_b_desc(b, n, k,
                                                         config.storage_config.load_block_n,
                                                         config.layout.block_k,
                                                         static_cast<int>(b.stride(0)),
                                                         config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.storage_config.store_block_m,
                                                config.storage_config.store_block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(
        config.layout.block_m, config.layout.block_n, MmaKind::MXFP4);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 sf_block_m, 32, 1, 0, 0, false, 2);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 sf_block_n, 32, 1, 0, 0, false, 2);

    const SM100MXFP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = epilogue_type,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd
    };
    const auto code = SM100MXFP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_mxfp4_gemm_1d1d", code);
    SM100MXFP4Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
