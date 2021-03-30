#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>

using namespace tensorflow;

REGISTER_OP("TfSwap")
.Input("matrix: float")
.Input("coordinate1: int32")
.Input("coordinate2: int32")
.Output("swapped_matrix: float");

class TfSwapOp : public OpKernel {
public:
	explicit TfSwapOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
    	// Grab the input tensor
    	Tensor matrix = context->input(0);
    	Tensor coordinate1 = context->input(1);
		Tensor coordinate2 = context->input(2);
		int r = matrix.shape().dim_size(0);
		int c = matrix.shape().dim_size(1);
		int i1 = coordinate1.tensor<int, 1>()(0);
		int j1 = coordinate1.tensor<int, 1>()(1);
		int i2 = coordinate2.tensor<int, 1>()(0);
		int j2 = coordinate2.tensor<int, 1>()(1);
		Tensor* swapped_matrxi = NULL;
		TensorShape output_shape = matrix.shape();
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &swapped_matrxi));
		auto input_ptr = matrix.flat<float>().data();
		auto output_ptr = swapped_matrxi->flat<float>().data();
		for(int i = 0; i < r * c; ++i){
			output_ptr[i] = input_ptr[i];
		}
		output_ptr[i1 * r + j1] = input_ptr[i2 * r +j2];
		output_ptr[i2 * r + j2] = input_ptr[i1 * r +j1];
	}
};

REGISTER_KERNEL_BUILDER(Name("TfSwap").Device(DEVICE_CPU), TfSwapOp);
REGISTER_KERNEL_BUILDER(Name("TfSwap").Device(DEVICE_GPU), TfSwapOp);

