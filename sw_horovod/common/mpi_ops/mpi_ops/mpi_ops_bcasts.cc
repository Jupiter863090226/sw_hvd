#include "mpi_ops.h"
//#include <stdio.h>

using namespace tensorflow;

REGISTER_OP("TfBroadcasts")
		.Attr("T: list({uint8, int8, uint16, int16, int32, int64, float32, float64})")
        .Input("values: T")
        .Output("output: T")
		.Attr("root: int")
        .SetShapeFn(
            [](::tensorflow::shape_inference::InferenceContext* c) {

	        int start_value_index = 0;
                int end_value_index = c->num_inputs();
                int num_outputs = c->num_outputs();
                int num_inputs = c->num_inputs();
                if (num_outputs != num_inputs) {
                    return errors::InvalidArgument("Tensor length of output should be equal to input");
                }
                for (int i = start_value_index; i < end_value_index; ++i){
                    c->set_output(i, c->input(i));
                }
                return Status::OK();
            }
        );


class TfBroadcastsOp : public OpKernel {
public:
    explicit TfBroadcastsOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, InputRange("values", &values_input_start_index_, &values_input_end_index_));
	OP_REQUIRES_OK(context, context->GetAttr("root", &root_));
	//printf("start build OP\n");
    }
    void Compute(OpKernelContext* context) override {

	//printf("start compute OP\n");

	// Grab the input tensor
	for (int i = values_input_start_index_; i < values_input_end_index_; ++i){
		Tensor input_tensor = context->input(i);
		//context->forward_ref_input_to_ref_output(0, 0);

		MPI_Datatype datatype = GetMPIDataType(input_tensor);

		MPI_Bcast(
				(void*)input_tensor.tensor_data().data(),
				(int)input_tensor.NumElements(),
				datatype,
				root_,
				MPI_COMM_WORLD);

		context->set_output(i, input_tensor);
}
    }

private:
  int root_;
  int values_input_start_index_;
  int values_input_end_index_;
};


REGISTER_KERNEL_BUILDER(Name("TfBroadcasts").Device(DEVICE_CPU), TfBroadcastsOp);
REGISTER_KERNEL_BUILDER(Name("TfBroadcasts").Device(DEVICE_GPU), TfBroadcastsOp);
