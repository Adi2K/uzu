use std::{
    cell::RefCell,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use thiserror::Error;

use super::{Linear, LinearBlockError, QuantizedLinear};
use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
        kernel::{
            HadamardTransformKernel, Kernels,
            quant_matmul::{QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernelEncodable},
        },
    },
    config::LinearConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum RHTLinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[source] Box<LinearBlockError<B>>),
    #[error("Parameter loading error: {0}")]
    ParameterError(ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Input dimension {input_dimension} is not divisible by block size {block_size}")]
    InputDimensionNotDivisibleByBlockSize {
        input_dimension: usize,
        block_size: usize,
    },
    #[error("Output dimension {output_dimension} is not divisible by block size {block_size}")]
    OutputDimensionNotDivisibleByBlockSize {
        output_dimension: usize,
        block_size: usize,
    },
    #[error("Input factors shape mismatch: expected [{expected_dimension}], got {actual_shape:?}")]
    InputFactorsShapeMismatch {
        expected_dimension: usize,
        actual_shape: Box<[usize]>,
    },
    #[error("Output factors shape mismatch: expected [{expected_dimension}], got {actual_shape:?}")]
    OutputFactorsShapeMismatch {
        expected_dimension: usize,
        actual_shape: Box<[usize]>,
    },
}

struct FusedOutputHadamardMatmul<B: Backend> {
    kernel: QuantizedMatmulKernelEncodable<B>,
    weights_buffer: Rc<RefCell<B::Buffer>>,
    scales_buffer: Rc<RefCell<B::Buffer>>,
    zero_points_or_biases_buffer: Rc<RefCell<B::Buffer>>,
}

pub struct RHTLinearWrapper<B: Backend> {
    inner_linear: Box<dyn Linear<B>>,
    input_hadamard: Option<(<B::Kernels as Kernels>::HadamardTransformKernel, B::Buffer)>,
    output_hadamard_kernel: <B::Kernels as Kernels>::HadamardTransformKernel,
    output_factors_buffer: B::Buffer,
    input_dimension: usize,
    output_dimension: usize,
    input_array_id: ArrayId,
    output_array_id: ArrayId,
    fused_output_hadamard: Option<FusedOutputHadamardMatmul<B>>,
}

fn try_create_fused_output_hadamard<B: Backend>(
    context: &B::Context,
    inner_config: &LinearConfig,
    quantized_linear: &QuantizedLinear<B>,
    input_dimension: usize,
    output_dimension: usize,
) -> Option<FusedOutputHadamardMatmul<B>> {
    let quant_config = match inner_config {
        LinearConfig::Quantized(q) | LinearConfig::MLXQuantized(q) => q,
        _ => return None,
    };

    if output_dimension % 32 != 0 {
        return None;
    }

    let kernel = QuantizedMatmulKernelEncodable::new(
        context,
        QuantizedMatmulConfiguration {
            data_type: quant_config.activation_precision.into(),
            group_size: quant_config.group_size,
            input_dim: input_dimension,
            output_dim: output_dimension,
            mode: quant_config.weight_quantization_mode,
            quantization_type: quantized_linear.quantization_type(),
            use_hadamard: true,
        },
    )
    .ok()?;

    Some(FusedOutputHadamardMatmul {
        kernel,
        weights_buffer: Rc::clone(quantized_linear.weights_buffer()),
        scales_buffer: Rc::clone(quantized_linear.scales_buffer()),
        zero_points_or_biases_buffer: Rc::clone(quantized_linear.zero_points_or_biases_buffer()),
    })
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        block_size: usize,
        inner_config: &LinearConfig,
        input_dimension: usize,
        output_dimension: usize,
        parameter_tree: &ParameterTree<B::Context>,
        input_array_id: ArrayId,
        output_array_id: ArrayId,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        if input_dimension % block_size != 0 {
            return Err(RHTLinearWrapperError::InputDimensionNotDivisibleByBlockSize {
                input_dimension,
                block_size,
            });
        }
        if output_dimension % block_size != 0 {
            return Err(RHTLinearWrapperError::OutputDimensionNotDivisibleByBlockSize {
                output_dimension,
                block_size,
            });
        }

        let kernel_data_type: DataType = inner_config.activation_precision().into();

        let input_factors_leaf = parameter_tree.leaf("input_factors").map_err(RHTLinearWrapperError::ParameterError)?;

        if input_factors_leaf.shape() != [input_dimension] {
            return Err(RHTLinearWrapperError::InputFactorsShapeMismatch {
                expected_dimension: input_dimension,
                actual_shape: input_factors_leaf.shape().into(),
            });
        }

        let output_factors_leaf =
            parameter_tree.leaf("output_factors").map_err(RHTLinearWrapperError::ParameterError)?;

        if output_factors_leaf.shape() != [output_dimension] {
            return Err(RHTLinearWrapperError::OutputFactorsShapeMismatch {
                expected_dimension: output_dimension,
                actual_shape: output_factors_leaf.shape().into(),
            });
        }

        let input_factors_buffer = input_factors_leaf.read_buffer().map_err(RHTLinearWrapperError::ParameterError)?;
        let output_factors_buffer = output_factors_leaf.read_buffer().map_err(RHTLinearWrapperError::ParameterError)?;

        let input_hadamard_kernel = <B::Kernels as Kernels>::HadamardTransformKernel::new(context, kernel_data_type)
            .map_err(RHTLinearWrapperError::BackendError)?;

        let output_hadamard_kernel = <B::Kernels as Kernels>::HadamardTransformKernel::new(context, kernel_data_type)
            .map_err(RHTLinearWrapperError::BackendError)?;

        let inner_linear_tree =
            parameter_tree.subtree("inner_linear").map_err(RHTLinearWrapperError::ParameterError)?;

        let (inner_linear, fused_output_hadamard) = match inner_config {
            LinearConfig::Quantized(q) | LinearConfig::MLXQuantized(q) => {
                let ql = QuantizedLinear::new(
                    context,
                    q,
                    input_dimension,
                    output_dimension,
                    &inner_linear_tree,
                    input_array_id,
                    output_array_id,
                )
                .map_err(|e| {
                    RHTLinearWrapperError::InnerLinearError(Box::new(super::LinearBlockError::QuantizedLinearError(e)))
                })?;

                let fused =
                    try_create_fused_output_hadamard(context, inner_config, &ql, input_dimension, output_dimension);

                (Box::new(ql) as Box<dyn Linear<B>>, fused)
            },
            _ => {
                let inner = <dyn Linear<B>>::new(
                    inner_config,
                    false,
                    input_dimension,
                    [output_dimension],
                    context,
                    &inner_linear_tree,
                    input_array_id,
                    output_array_id,
                )
                .map_err(|error| RHTLinearWrapperError::InnerLinearError(Box::new(error)))?;
                (inner, None)
            },
        };

        Ok(Self {
            inner_linear,
            input_hadamard: Some((input_hadamard_kernel, input_factors_buffer)),
            output_hadamard_kernel,
            output_factors_buffer,
            input_dimension,
            output_dimension,
            input_array_id,
            output_array_id,
            fused_output_hadamard,
        })
    }

    pub fn take_input_hadamard_factors(&mut self) -> Option<B::Buffer> {
        self.input_hadamard.take().map(|(_, factors)| factors)
    }
}

impl<B: Backend> Linear<B> for RHTLinearWrapper<B> {
    fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let batch_size = state.active_row_count();

        // Input Hadamard (standalone dispatch, or skipped if fused into preceding norm)
        if let Some((ref kernel, ref factors_buffer)) = self.input_hadamard {
            let input_array = state.array(self.input_array_id);
            kernel.encode(
                input_array.buffer().borrow_mut().deref_mut(),
                factors_buffer,
                self.input_dimension as u32,
                batch_size as u32,
                encoder,
            );
        }

        // Fused matmul + output Hadamard (handles both decode and prefill batch sizes)
        if let Some(ref fused) = self.fused_output_hadamard {
            let input_array = state.array(self.input_array_id);
            let output_array = state.array(self.output_array_id);
            let in_buf_rc = input_array.buffer();
            let out_buf_rc = output_array.buffer();

            fused
                .kernel
                .encode(
                    encoder,
                    QuantizedMatmulArguments {
                        a_buffer: in_buf_rc.borrow().deref(),
                        a_offset: 0,
                        b_buffer: fused.weights_buffer.borrow().deref(),
                        scales_buffer: fused.scales_buffer.borrow().deref(),
                        zero_points_or_biases_buffer: fused.zero_points_or_biases_buffer.borrow().deref(),
                        output_buffer: out_buf_rc.borrow_mut().deref_mut(),
                        hadamard_factors: Some(&self.output_factors_buffer),
                        batch_dim: batch_size,
                    },
                )
                .expect("Fused output hadamard matmul encode failed");

            return Ok(());
        }

        // Fallback: separate inner_linear + output Hadamard
        self.inner_linear.encode(state, encoder)?;

        {
            let output_array = state.array(self.output_array_id);
            self.output_hadamard_kernel.encode(
                output_array.buffer().borrow_mut().deref_mut(),
                &self.output_factors_buffer,
                self.output_dimension as u32,
                batch_size as u32,
                encoder,
            );
        }

        Ok(())
    }
}
