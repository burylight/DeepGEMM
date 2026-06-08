#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          typename cd_dtype_t,
          typename epilogue_type_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_nvfp4_gemm_1d1d_impl(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "NVFP4 GEMM only supports BF16 output");

    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 64;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(BLOCK_M == 128, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N % 16 == 0 and 16 <= BLOCK_N and BLOCK_N <= 256, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 256, "Invalid block K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT(kNumMulticast == 1 or not kIsMulticastOnA, "NVFP4 2SM only supports M clustering");
    DG_STATIC_ASSERT(kSwizzleAMode == 128 and kSwizzleBMode == 128, "NVFP4 operands require 128B swizzle");

    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumSFWordsPerStage = BLOCK_K / 64;
    DG_STATIC_ASSERT(kNumSFWordsPerStage == 4, "NVFP4 block K requires four packed scale columns");

    constexpr uint32_t kNumEpilogueStages = BLOCK_N == 256 ? 1 : 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K / 2;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K / 2;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * kNumSFWordsPerStage * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * kNumSFWordsPerStage * sizeof(uint32_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = kNumSFWordsPerStage * SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = kNumSFWordsPerStage * SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : void();

    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint8_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    auto sf_start_ptr = reinterpret_cast<uint8_t*>(smem_b[kNumStages]);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
    auto tmem_ptr_in_smem  = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(kNumMulticast * 64);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    cudaGridDependencySynchronize();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, 1, kNumMulticast, kIsMulticastOnA, kNumSMs, 64>(
        shape_m, shape_n, shape_k, nullptr);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx == 0 and cute::elect_one_sync()) {
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                uint32_t m_idx = m_block_idx * BLOCK_M;
                uint32_t n_idx = n_block_idx * BLOCK_N;
                uint32_t m_load_idx = m_idx;
                uint32_t n_load_idx = n_idx;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                if constexpr (kNumMulticast > 1) {
                    m_load_idx += kIsMulticastOnA ? cute::block_rank_in_cluster() * LOAD_BLOCK_M : 0;
                    n_load_idx += kIsMulticastOnA ? 0 : cute::block_rank_in_cluster() * LOAD_BLOCK_N;
                }

                tma::copy_packed_fp4<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_load_idx);
                tma::copy_packed_fp4<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_load_idx);

                uint32_t sf_k_idx = math::ceil_div(k_idx, 64u);
                tma::copy<SF_BLOCK_M, kNumSFWordsPerStage, 0>(
                    &tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], m_idx, sf_k_idx);
                tma::copy<SF_BLOCK_N, kNumSFWordsPerStage, 0>(
                    &tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx], n_idx, sf_k_idx);

                constexpr uint32_t num_arrival_bytes =
                    SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                    SF_BLOCK_M * kNumSFWordsPerStage * sizeof(uint32_t) +
                    SF_BLOCK_N * kNumSFWordsPerStage * sizeof(uint32_t);
                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
            cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
            UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_smem_desc(
            cute::UMMA::LayoutType::SWIZZLE_128B, smem_a[0], 8 * (BLOCK_K / 2), 0);
        auto b_desc = mma::sm100::make_smem_desc(
            cute::UMMA::LayoutType::SWIZZLE_128B, smem_b[0], 8 * (BLOCK_K / 2), 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        DG_STATIC_ASSERT(((UMMA_M == 128 or UMMA_M == 256) and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                auto umma_arrive = [](const uint64_t* barrier) {
                    if constexpr (kNumMulticast == 1) {
                        cutlass::arch::umma_arrive(barrier);
                    } else {
                        constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    }
                };
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                with_sf_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                if (cute::elect_one_sync()) {
                    using cute_utccp_t = cute::conditional_t<
                        kNumMulticast == 1, cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                    #pragma unroll
                    for (uint32_t sf_word_col = 0; sf_word_col < kNumSFWordsPerStage; ++ sf_word_col) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + sf_word_col * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(
                                sf_desc, kTmemStartColOfSFA + sf_word_col * (SF_BLOCK_M / 32) + i * 4);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + sf_word_col * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(
                                sf_desc, kTmemStartColOfSFB + sf_word_col * (SF_BLOCK_N / 32) + i * 4);
                        }
                    }

                    #pragma unroll
                    for (uint32_t k64_idx = 0; k64_idx < BLOCK_K / UMMA_K; ++ k64_idx) {
                        const uint32_t sf_word_col = k64_idx;
                        const auto runtime_instr_desc =
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, 0, 0);

                        a_desc.lo = a_desc_base_lo + (k64_idx * (UMMA_K / 2)) / 16;
                        b_desc.lo = b_desc_base_lo + (k64_idx * (UMMA_K / 2)) / 16;
                        using mma_t = cute::conditional_t<
                            kNumMulticast == 1, ptx::SM100_MMA_MXF4NVF4_SS, ptx::SM100_MMA_MXF4NVF4_2x1SM_SS>;
                        mma_t::fma(
                            a_desc, b_desc, accum_stage_idx * UMMA_N,
                            k_block_idx > 0 or k64_idx > 0, runtime_instr_desc,
                            kTmemStartColOfSFA + sf_word_col * (SF_BLOCK_M / 32),
                            kTmemStartColOfSFB + sf_word_col * (SF_BLOCK_N / 32));
                    }
                }
                __syncwarp();

                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }
    } else if (warp_idx == 2 or warp_idx == 3) {
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);

                #pragma unroll
                for (uint32_t sf_word_col = 0; sf_word_col < kNumSFWordsPerStage; ++ sf_word_col) {
                    if (warp_idx == 2) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                            utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + sf_word_col * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                    } else {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                            utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + sf_word_col * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                    }
                }

                cutlass::arch::fence_view_async_shared();
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        uint32_t tma_stage_idx = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = m_block_idx * BLOCK_M;
            const auto base_n_idx = n_block_idx * BLOCK_N;

            epilogue::sm100_store_cd<
                BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                GemmType::Normal, false,
                cd_dtype_t, epilogue_type_t>
            (smem_cd, tma_stage_idx, tmem_base_addr,
             base_m_idx, base_n_idx, 0,
             epilogue_warp_idx, lane_idx,
             tmem_empty_barriers[accum_stage_idx],
             tensor_map_cd);
        }
    }

    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
