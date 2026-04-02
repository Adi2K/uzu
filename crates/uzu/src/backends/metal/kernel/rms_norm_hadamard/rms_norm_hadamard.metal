#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 4
#define STAGING_SIZE (BLOCK_SIZE * GRAIN_SIZE * 2)

template <typename InputT, typename ScaleT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, half, bfloat)
VARIANTS(ScaleT, float, half, bfloat)
VARIANTS(OutputT, float, half, bfloat)
VARIANTS(AccumT, float, half)
PUBLIC KERNEL(RMSNormHadamardMul)(
    const device InputT* input OPTIONAL(!in_place),
    const device ScaleT* scales,
    device OutputT* output,
    device InputT* shortcut_buffer OPTIONAL(copy_to_shortcut),
    const device int32_t* hadamard_factors,
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    constant bool& full_layer,
    const bool in_place SPECIALIZE,
    const bool copy_to_shortcut SPECIALIZE,
    const bool residual_add SPECIALIZE,
    threadgroup float staging[STAGING_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(1024)
) {
  if (in_place) {
    input = reinterpret_cast<const device InputT*>(output);
  }

  threadgroup AccumT* shared_sum = reinterpret_cast<threadgroup AccumT*>(
      &staging[STAGING_SIZE - METAL_SIMD_SIZE]
  );

  const uint input_offset = batch_idx * element_count;
  const device InputT* input_data = input + input_offset;
  const device ScaleT* scales_data = scales;
  device OutputT* output_data = output + input_offset;

  // ── Phase 1: RMSNorm reduction + cache to staging ──────────────────

  AccumT partial_sum = static_cast<AccumT>(0.0f);

  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i < element_count) {
        InputT val = input_data[i];
        if (copy_to_shortcut) {
          if (residual_add) {
            val = val + shortcut_buffer[input_offset + i];
          }
          shortcut_buffer[input_offset + i] = val;
        }
        vals[j] = static_cast<AccumT>(val);
      } else {
        vals[j] = 0.0f;
      }
      if (i < element_count)
        staging[i] = float(vals[j]);
    }
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      partial_sum += vals[j] * vals[j];
    }
  }

  AccumT total_sum =
      threadgroup_cooperative_reduce<SimdReduceSum<AccumT>, BLOCK_SIZE>(
          partial_sum,
          shared_sum,
          thread_context
      );

  AccumT mean_square =
      static_cast<AccumT>(total_sum) / static_cast<AccumT>(element_count);
  AccumT rms_norm = rsqrt(mean_square + static_cast<AccumT>(epsilon));

  // ── Phase 1b: Normalize + scale, write to staging ──────────────────

  for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count;
       base_i += BLOCK_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(staging[i]) : 0.0f;
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count)
        continue;

      AccumT normalized_high = vals[j] * rms_norm;
      float result;

      if (full_layer) {
        AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) +
                                  static_cast<AccumT>(scale_offset);
        result = float(normalized_high * scale_value_high);
      } else {
        OutputT normalized_low = static_cast<OutputT>(normalized_high);
        OutputT scale_value_low = static_cast<OutputT>(
            static_cast<AccumT>(scales_data[i]) +
            static_cast<AccumT>(scale_offset)
        );
        result = float(normalized_low * scale_value_low);
      }

      staging[i] = result;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ── Phase 2: Hadamard transform from staging to device ─────────────

  const uint lane = thread_in_row % METAL_SIMD_SIZE;
  const uint simd_group_id = thread_in_row / METAL_SIMD_SIZE;
  const uint total_simd_groups = BLOCK_SIZE / METAL_SIMD_SIZE;
  const uint total_blocks = element_count / METAL_SIMD_SIZE;

  for (uint block = simd_group_id; block < total_blocks;
       block += total_simd_groups) {
    uint elem_idx = block * METAL_SIMD_SIZE + lane;
    output_data[elem_idx] = OutputT(simdgroup_random_hadamard_transform(
        static_cast<ushort>(lane),
        staging[elem_idx],
        hadamard_factors[elem_idx]
    ));
  }
}
