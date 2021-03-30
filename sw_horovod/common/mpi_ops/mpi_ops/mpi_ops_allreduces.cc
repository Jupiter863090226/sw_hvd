#include "mpi_ops.h"
#include <stdio.h>

using namespace tensorflow;

REGISTER_OP("TfAllreduces")
        .Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
        .Input("values: N * T")
        .Output("output: N * T")
	.Attr("precision: int")
        .Attr("N: int >= 1")
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


class TfAllreducesOp : public OpKernel {
public:
    explicit TfAllreducesOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, InputRange("values", &values_input_start_index_, &values_input_end_index_));
	OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));
	//printf("start build OP\n");
    }
    void Compute(OpKernelContext* context) override {

	//printf("start compute OP\n");
	if (0){
	    
	    int buf_size = 0;
	    for (int i = values_input_start_index_; i < values_input_end_index_; ++i)
		buf_size += context->input(i).NumElements();
	    
	    if (precision_ == 0){ //precision_: precision used for data buffer and MPI_OP (0 for float and 1 for double)
	        Tensor buf_tensor;
	   		DataType prec_type = tensorflow::DT_FLOAT;
	        OP_REQUIRES_OK(context, context->allocate_temp(prec_type, TensorShape({buf_size}), &buf_tensor));
	    	auto buf_ptr = buf_tensor.flat<float>().data();
	    	
	    	int buf_len = 0;
	    	for (int i = values_input_start_index_; i < values_input_end_index_; ++i){
	    	    Tensor input_tensor = context->input(i);

	    	    if (input_tensor.dtype() == tensorflow::DT_FLOAT){
	    	        auto input_len = input_tensor.flat<float>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<float>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    buf_ptr[buf_len+ii] = input_ptr[ii];
	    	        
	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_DOUBLE){
	    	        auto input_len = input_tensor.flat<double>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<double>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    buf_ptr[buf_len+ii] = (float)input_ptr[ii];

	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_INT32){
	    	        auto input_len = input_tensor.flat<int32>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<int32>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    buf_ptr[buf_len+ii] = input_ptr[ii];

	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_INT64){
	    	        auto input_len = input_tensor.flat<int64>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<int64>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    buf_ptr[buf_len+ii] = input_ptr[ii];

	    	        buf_len += input_len;
	    	    } else {
	    	        std::cout << "Error for type " << input_tensor.dtype() << " of Tensor: " << input_tensor.DebugString() << std::endl;
	    	    }
	    	}
	    	assert(buf_len == buf_size);
	    
		//std::cout << "buf_type: float, buf_len: " << buf_len << std::endl;

		//std::cout << "Before MPI_OP: " << std::endl;
	    	//for(int ii=0; ii<buf_size; ii++)
		//    printf("buf_idx: %d, buf_value: %f", ii, *(buf_ptr+ii));
		//std::cout << std::endl;

		MPI_Datatype datatype = GetMPIDataType(buf_tensor);
            	MPI_Allreduce(
            	        MPI_IN_PLACE,
            	        (void*)buf_ptr,//buf_tensor.tensor_data().data(),
            	        buf_size,//buf_tensor.NumElements(),
            	        datatype,
            	        MPI_SUM,
            	        MPI_COMM_WORLD);
	    	
		//std::cout << "After MPI_OP " << std::endl;
	    	//for(int ii=0; ii<buf_size; ii++)
		//    printf("buf_idx: %d, buf_value: %f", ii, *(buf_ptr+ii));
		//std::cout << std::endl;

	    	buf_len = 0;
	    	for (int i = values_input_start_index_; i < values_input_end_index_; ++i){
	    	    Tensor input_tensor = context->input(i);

	    	    if (input_tensor.dtype() == tensorflow::DT_FLOAT){
	    	        auto input_len = input_tensor.flat<float>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<float>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	            input_ptr[ii] = buf_ptr[buf_len+ii];
	    	        
	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_DOUBLE){
	    	        auto input_len = input_tensor.flat<double>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<double>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    input_ptr[ii] = buf_ptr[buf_len+ii];
	    	        
	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_INT32){
	    	        auto input_len = input_tensor.flat<int32>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<int32>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    input_ptr[ii] = (int)buf_ptr[buf_len+ii];

	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_INT64){
	    	        auto input_len = input_tensor.flat<int64>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<int64>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    input_ptr[ii] = (int)buf_ptr[buf_len+ii];

	    	        buf_len += input_len;
	    	    } else {
	    	        std::cout << "Error for type " << input_tensor.dtype() << " of Tensor: " << input_tensor.DebugString() << std::endl;
	    	    }
                    
		    context->set_output(i, input_tensor);
	    	}
	    	assert(buf_len == buf_size);
	    } else {
	        Tensor buf_tensor;
	    	DataType prec_type = tensorflow::DT_DOUBLE;
	    	OP_REQUIRES_OK(context, context->allocate_temp(prec_type, TensorShape({buf_size}), &buf_tensor));
	    	auto buf_ptr = buf_tensor.flat<double>().data();
		
	    	int buf_len = 0;
	    	for (int i = values_input_start_index_; i < values_input_end_index_; ++i){
	    	    Tensor input_tensor = context->input(i);

	    	    if (input_tensor.dtype() == tensorflow::DT_FLOAT){
	    	        auto input_len = input_tensor.flat<float>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<float>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    buf_ptr[buf_len+ii] = input_ptr[ii];
	    	        
	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_DOUBLE){
	    	        auto input_len = input_tensor.flat<double>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<double>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    buf_ptr[buf_len+ii] = input_ptr[ii];

	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_INT32){
	    	        auto input_len = input_tensor.flat<int32>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<int32>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    buf_ptr[buf_len+ii] = input_ptr[ii];

	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_INT64){
	    	        auto input_len = input_tensor.flat<int64>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<int64>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    buf_ptr[buf_len+ii] = input_ptr[ii];

	    	        buf_len += input_len;
	    	    } else {
	    	        std::cout << "Error for type " << input_tensor.dtype() << " of Tensor: " << input_tensor.DebugString() << std::endl;
	    	    }
	    	}
	    	assert(buf_len == buf_size);
	    
	    	MPI_Datatype datatype = GetMPIDataType(buf_tensor);
            	MPI_Allreduce(
            	        MPI_IN_PLACE,
            	        (void*)buf_ptr,//buf_tensor.tensor_data().data(),
            	        buf_size,//buf_tensor.NumElements(),
            	        datatype,
            	        MPI_SUM,
            	        MPI_COMM_WORLD);
	    	
	    	buf_len = 0;
	    	for (int i = values_input_start_index_; i < values_input_end_index_; ++i){
	    	    Tensor input_tensor = context->input(i);

	    	    if (input_tensor.dtype() == tensorflow::DT_FLOAT){
	    	        auto input_len = input_tensor.flat<float>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<float>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	            input_ptr[ii] = (float)buf_ptr[buf_len+ii];
	    	        
	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_DOUBLE){
	    	        auto input_len = input_tensor.flat<double>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<double>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    input_ptr[ii] = buf_ptr[buf_len+ii];
	    	        
	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_INT32){
	    	        auto input_len = input_tensor.flat<int32>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<int32>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    input_ptr[ii] = (int)buf_ptr[buf_len+ii];

	    	        buf_len += input_len;
	    	    } else if (input_tensor.dtype() == tensorflow::DT_INT64){
	    	        auto input_len = input_tensor.flat<int64>().size();
	    	        assert(input_len == input_tensor.NumElements());
	    	        auto input_ptr = input_tensor.flat<int64>().data();
	    	        for (int ii = 0; ii < input_len; ii++)
	    	    	    input_ptr[ii] = (int)buf_ptr[buf_len+ii];

	    	        buf_len += input_len;
	    	    } else {
	    	        std::cout << "Error for type " << input_tensor.dtype() << " of Tensor: " << input_tensor.DebugString() << std::endl;
	    	    }
                    
		    context->set_output(i, input_tensor);
	    	}
	    	assert(buf_len == buf_size);
	    }

	} else {
            // Grab the input tensor
            for (int i = values_input_start_index_; i < values_input_end_index_; ++i){
                Tensor input_tensor = context->input(i);
                //context->forward_ref_input_to_ref_output(0, 0);

                MPI_Datatype datatype = GetMPIDataType(input_tensor);

                MPI_Allreduce(
                        MPI_IN_PLACE,
                        (void*)input_tensor.tensor_data().data(),
                        (int)input_tensor.NumElements(),
                        datatype,
                        MPI_SUM,
                        MPI_COMM_WORLD);

                context->set_output(i, input_tensor);
	    	}
        }
    }

private:
  int precision_;
  int values_input_start_index_;
  int values_input_end_index_;
};


REGISTER_KERNEL_BUILDER(Name("TfAllreduces").Device(DEVICE_CPU), TfAllreducesOp);
REGISTER_KERNEL_BUILDER(Name("TfAllreduces").Device(DEVICE_GPU), TfAllreducesOp);
