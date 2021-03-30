#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>

using namespace tensorflow;

REGISTER_OP("TfExchange")
.Input("matrix_a:int32")
.Input("matrix_b:int32")
.Input("matrix_c:int32")
.Input("index_list:int32")
.Input("direction:int32")
.Output("batches:int32");

class TfExchangeOp : public OpKernel {
public:
	explicit TfExchangeOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override {
    	// Grab the input tensor
		
    	Tensor A = context->input(0);
    	Tensor B = context->input(1);
		Tensor C = context->input(2);
		Tensor index_list = context->input(3);
		Tensor direction = context->input(4);
		auto A_mapped = A.tensor<int, 2>();
		auto B_mapped = B.tensor<int, 2>();
		auto C_mapped = C.tensor<int, 2>();
		auto index_list_mapped = index_list.tensor<int, 2>();
		auto direction_mapped = direction.tensor<int, 1>();

		int r = A.shape().dim_size(0);
		int c = A.shape().dim_size(1);
		int batches = index_list.shape().dim_size(0);
		Tensor* output_tensor = NULL;
		TensorShape output_shape = TensorShape({batches, r, c});
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
		auto out_tensor_mapped = output_tensor->tensor<int, 3>();
		// copy A to output tensor
		for(int i = 0; i < batches; ++i){
			for(int j = 0; j < r; ++j){
				for(int k = 0; k < c; ++k){
					out_tensor_mapped(i, j, k) = A_mapped(j, k);
				}
			}
		}
		int dir_r = direction_mapped(0);
		int dir_c = direction_mapped(1);
		// exchange
		for(int i = 0; i < batches; ++i){
			int idx_r = index_list_mapped(i, 0);
			int idx_c = index_list_mapped(i, 1);
			out_tensor_mapped(i, idx_r, idx_c) = B_mapped(idx_r, idx_c);
			out_tensor_mapped(i, (idx_r + dir_r) % r, (idx_c + dir_c) % c) = C_mapped((idx_r + dir_r) % r, (idx_c + dir_c) % c);
		}
	}
};

REGISTER_KERNEL_BUILDER(Name("TfExchange").Device(DEVICE_CPU), TfExchangeOp);
REGISTER_KERNEL_BUILDER(Name("TfExchange").Device(DEVICE_GPU), TfExchangeOp);

