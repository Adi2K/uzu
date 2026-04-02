use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(RMSNormHadamardMul)]
#[variants(InputT, f32, f16, bf16)]
#[variants(ScaleT, f32, f16, bf16)]
#[variants(OutputT, f32, f16, bf16)]
#[variants(AccumT, f32, f16)]
pub fn rms_norm_hadamard_mul<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    #[optional(!in_place)]
    #[allow(unused)]
    input: Option<*const InputT>,
    #[allow(unused)] scales: *const ScaleT,
    #[allow(unused)] output: *mut OutputT,
    #[optional(copy_to_shortcut)] shortcut_buffer: Option<*mut InputT>,
    #[allow(unused)] hadamard_factors: *const i32,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] element_count: u32,
    #[allow(unused)] epsilon: f32,
    #[allow(unused)] scale_offset: f32,
    #[allow(unused)] full_layer: bool,
    #[specialize]
    #[allow(unused)]
    in_place: bool,
    #[specialize] copy_to_shortcut: bool,
    #[specialize] residual_add: bool,
) {
    todo!()
}
