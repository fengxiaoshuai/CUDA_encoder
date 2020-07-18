#include "encoder.h"
#include <thrust/device_vector.h>
#define FINAL_MASK 0xffffffff
namespace fastertransformer
{

/******************************************************device********************************************************/
template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}
template <typename T>
__inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}



template <typename T>
__inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32]; 
//  __shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : 0;
  val = warpReduceMax(val);

  return val;
}


template <typename T>
  __inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32]; 
  //__shared__ T shared[32]; 
  int lane = threadIdx.x & 0x1f; 
  int wid = threadIdx.x >> 5;  

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
                              
  return val;
}

template <typename T>
__global__ 
void add_bias_kernel(T* output,  const T* bias, const int m, const int n)
{
  int id = blockIdx.x * n + threadIdx.x;
  output[id] = output[id]  + __ldg(&bias[threadIdx.x]);
}


template <typename T>
__global__
void softmax_kernel(T* qk_buf, T* softmax_buf, const int batch_size, const int head_num, const int length)
{
    int qk_offset = blockIdx.x * length * length;
    __shared__ float s_sum, s_max;

    for(int i = 0; i < length; ++i)
    {
      float qk = threadIdx.x < length ? (float)qk_buf[threadIdx.x + qk_offset] : 0.0f;
      float tmp = threadIdx.x < length ? (float)(qk ): -1e20f;
      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0) s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < length ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0) s_sum = sum_val + 1e-6f;
      __syncthreads();

      if(threadIdx.x < length) softmax_buf[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += length;
    }
}

template <typename T>
__global__
void preprocess_kernel(const T* input, T* output, const T* gamma, const T* beta, int m, int n)
{
  //m=batch*length, n=hidden_unit
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float local_out = tid < n ? (float)(__ldg(&input[blockIdx.x * n + tid])) : 0.0f;
  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
  {
    s_mean = mean / n;
  }
  __syncthreads();

  variance = blockReduceSum<float>(tid < n ? (local_out - s_mean) * (local_out - s_mean) : 0.0f);

  if(threadIdx.x == 0)
  {
    s_variance = rsqrtf(variance / n + 1e-6);
  }
  __syncthreads();

  if(tid < n)
  {
    output[blockIdx.x * n + tid] =  (T)(((local_out - s_mean) * s_variance) * \
    (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
  }
  __syncthreads();
}

template <typename T>
__global__ 
void add_input_kernel(T* output, const T* input, const int m, const int n)
{
  int id = blockIdx.x * n + threadIdx.x;
  output[id] = output[id] + input[id];
}

template <typename T>
__global__ 
void build_bias_kernel(T* out_tensor, int* mask)
{
  int write_pos = threadIdx.x + blockIdx.x * 1024;
  out_tensor[write_pos] +=  (1-mask[blockIdx.x]) * (T)-1e9;
}


template <typename T>
__global__ 
void set_zero_kernel(T* out_tensor, const int* mask, const int hidden_units)
{
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  out_tensor[write_pos] *=  mask[blockIdx.x];
}

template <typename T>
__global__ 
void encoder_embedding_lookup_kernel(const T* embedding_table, const T* embedding_language, const int* mask, const int* language, const int* input_ids, const int length, const int hidden_units, T* out_tensor)
{
  int write_pos = threadIdx.x + blockIdx.x * hidden_units;
  out_tensor[write_pos] = embedding_table[input_ids[blockIdx.x] * hidden_units + threadIdx.x] * (T)32.0f * mask[blockIdx.x];
  out_tensor[write_pos] = embedding_language[language[blockIdx.x/length] * hidden_units + threadIdx.x]  +  out_tensor[write_pos];
}

template <typename T>
__global__ 
void position_embedding_kernel(T* out_key, T* out_value, const T* key_table, const T* value_table, const int* mat)
{
  int write_pos = threadIdx.x + blockIdx.x * 64;
  out_key[write_pos] = key_table[mat[blockIdx.x] * 64 + threadIdx.x];
  out_value[write_pos] = value_table[mat[blockIdx.x] * 64 + threadIdx.x];
}


template <typename T>
__global__ 
void build_bias_kernel(const int* mask, T* bias)
{
	int pos = threadIdx.x;
	bias[pos] = (1-mask[pos]) * (T)-1e9f;
}


template <typename T>
__global__ 
void embedding_kernel(const T* k_table, const T* v_table, T* k_out, T* v_out,
                      int* position_ids, const int num)
{
  int write_pos = threadIdx.x + blockIdx.x * num;
  k_out[write_pos] = k_table[position_ids[blockIdx.x] * num + threadIdx.x];
  v_out[write_pos] = v_table[position_ids[blockIdx.x] * num + threadIdx.x];
}


template <typename T>
__global__ 
void add_bias_relu(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m)
    {
      val = out[tid + i * blockDim.x + row_id * n] + reg_bias;
      out[tid + i * blockDim.x + row_id * n] = (T)(val > (T)0.0f ? val : (T)0.0f);
      row_id += gridDim.x;
     }
  }
}

/******************************************************host********************************************************/


template <OperationType OpType_>
OpenEncoder<OpType_>::OpenEncoder(const IAllocator &allocator, const int& batch_size, const int& length,
				  const int& head_num, const int& size_per_head, const int& hidden_units,
 				  const int& memory_hidden_units, const int& layer_num, ParaMeter_* param,
             			  DataType_ * p_table_embedding, DataType_ * p_table_language,
           		          DataType_ * p_gamma, DataType_ * p_beta):allocator_ (allocator) 

{
	batch_size_= batch_size;
	length_ = length;
	head_num_ = head_num;
	size_per_head_ = size_per_head;
	hidden_units_ = hidden_units;
	layer_num_ = layer_num;
        param_ = param;
        p_table_embedding_ = p_table_embedding;
	p_table_language_ = p_table_language;
	p_gamma_ = p_gamma;
	p_beta_ = p_beta;

        if (Traits_::OpType == OperationType::FP32)
        {cublasAlgo_ = CUBLAS_GEMM_DEFAULT;}
        else
        {cublasAlgo_ = CUBLAS_GEMM_DEFAULT_TENSOR_OP;}
}



template <OperationType OpType_>
void OpenEncoder<OpType_>::initialize(DataType_* buf) 
{
    int mat_size = length_ * length_; 
    long long buf_size = batch_size_* length_ * hidden_units_;
    long long datatype_size = 12 * buf_size + 2 * size_per_head_ * length_ * length_ + batch_size_ * length_ + 2 * length_ * length_ * batch_size_ * head_num_;
    buf = reinterpret_cast<DataType_ *>(allocator_.malloc(sizeof(DataType_) * datatype_size + sizeof(int) * mat_size)); 

    input_tensor_buf_  = buf;
    output_tensor_buf_ = buf + 1 * buf_size; 
    q_tensor_buf_      = buf + 2 * buf_size; 
    k_tensor_buf_      = buf + 3 * buf_size;     
    v_tensor_buf_      = buf + 4 * buf_size; 
    ffn_out_buf_       = buf + 5 * buf_size; 
    encoder_result_buf_= buf + 6 * buf_size; 
    qv_buf_            = buf + 7 * buf_size; 
    qk_buf_            = buf + 8 * buf_size;
    bias_buf_          = qk_buf_ + batch_size_ * length_ * length_ * head_num_;
    softmax_buf_       = bias_buf_ + batch_size_ * length_;
    position_key_buf_  = softmax_buf_ + batch_size_ * length_ * length_ * head_num_;
    position_value_buf_= position_key_buf_ + length_ * length_ * 64;
    inner_tensor_buf_  = position_value_buf_ + length_ * length_ * 64;
    mat_buf_           = (int*)inner_tensor_buf_ + buf_size * 4;
                   
}



template <OperationType OpType_>
void OpenEncoder<OpType_>::embedding_lookup(const DataType_* embedding_table, const DataType_* embedding_language, const int* input_ids, DataType_* out_tensor, const int batch_size, const int length, const int* mask, const int* language)
{
	dim3 grid(batch_size * length);
	dim3 block(hidden_units_);
	encoder_embedding_lookup_kernel <DataType_> <<<grid, block, 0, param_[0].stream>>>(embedding_table, embedding_language, mask, language, input_ids, length,  hidden_units_, out_tensor);
}

template <OperationType OpType_>
void OpenEncoder<OpType_>::preprocess(const DataType_* input, DataType_* output, int length, int batch_size, int hidden_units, ParaMeter_ param) 
{
  dim3 grid(batch_size * length);
  dim3 block(hidden_units);
  preprocess_kernel <DataType_> <<<grid, block, 0, param.stream>>>(input, output, param.self_layernorm.gamma, param.self_layernorm.beta, batch_size*length, hidden_units);
}

template <OperationType OpType_>
void OpenEncoder<OpType_>::lastprocess(const DataType_* input, DataType_* output, int length, int batch_size, int hidden_units, const DataType_* gamma, const DataType_* beta) 
{
  dim3 grid(batch_size * length);
  dim3 block(hidden_units);
  preprocess_kernel <DataType_> <<<grid, block, 0, param_[0].stream>>>(input, output, gamma, beta, batch_size*length, hidden_units);
}


template<OperationType OpType_>
void OpenEncoder<OpType_>::encoder_attention(DataType_* input_tensor, DataType_ * query_buf, DataType_* key_buf, 
                                             DataType_* value_buf, DataType_* position_key, DataType_* position_value, 
                                             DataType_* qk_buf, DataType_ * qv_buf, ParaMeter_ param)
{

  int m = batch_size_ * length_;
  int n = hidden_units_;
  int k = hidden_units_;


  DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
  DataType_ scalar = (DataType_) 1 / sqrtf(size_per_head_ * 1.0f);
   //compute Qkv
   
   //compute Q
   check_cuda_error(cublasGemmEx(param.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &scalar, 
    param.self_attention.query_weight , AType_, n, 
    input_tensor, BType_, k, 
    &beta, 
    query_buf, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_)));
  //compute K
  check_cuda_error(cublasGemmEx(param.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param.self_attention.key_weight, AType_, n, 
    input_tensor, BType_, k, 
    &beta, 
    key_buf, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_)));
  //compute V
  check_cuda_error(cublasGemmEx(param.cublas_handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, 
    n, m, k, 
    &alpha, 
    param.self_attention.value_weight, AType_, n, 
    input_tensor, BType_, k, 
    &beta, 
    value_buf, CType_, n, 
    computeType_, 
    static_cast<cublasGemmAlgo_t>(cublasAlgo_)));

   // compute q/sqrt(64)
   //Q * position_key
    
    thrust::device_vector<float*> h_a_array(m * head_num_);
    thrust::device_vector<float*> h_b_array(m * head_num_);
    thrust::device_vector<float*> h_c_array(m * head_num_);
    for(int i=0; i<batch_size_; ++i)
    {
	for(int j=0; j<length_; j++)
	{
    		h_a_array[i * length_ + j] = query_buf + i * length_ * hidden_units_ + j * hidden_units_;
    		h_b_array[i * length_ + j] = position_key + j * (hidden_units_ / head_num_) * length_;
    		h_c_array[i * length_ + j] = qk_buf + i * length_ * (length_ * head_num_) + j * length_ * head_num_;
	}
    }

    const void **a_array, **b_array;
    void **c_array;
    cudaMalloc((void**)&a_array, m * sizeof(float *));
    cudaMalloc((void**)&b_array, m * sizeof(float *));
    cudaMalloc((void**)&c_array, m * sizeof(float *));
    cudaMemcpy(a_array, thrust::raw_pointer_cast(&h_a_array[0]), m * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(b_array, thrust::raw_pointer_cast(&h_b_array[0]), m * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(c_array, thrust::raw_pointer_cast(&h_c_array[0]), m * sizeof(float *), cudaMemcpyHostToDevice);

    
    int lda = (hidden_units_ / head_num_);
    int ldb = lda;
    int ldc = length_;
    int batchCount = batch_size_ * length_;
    
    check_cuda_error(cublasGemmBatchedEx(param.cublas_handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			length_,
			head_num_,
		        (hidden_units_/head_num_),
			&alpha,
		        b_array, BType_, ldb,
			a_array, AType_, lda,
			&beta,
			c_array, CType_, ldc,
			batchCount,
			computeType_,
			static_cast<cublasGemmAlgo_t>(cublasAlgo_)));
	
     
	
    beta = (DataType_)1.0f;
    //compute Q * K + qpk_buf_
    check_cuda_error(cublasGemmStridedBatchedEx(param.cublas_handle,
	CUBLAS_OP_T, CUBLAS_OP_N,
	length_, length_, size_per_head_,
	&alpha,
	key_buf, AType_, size_per_head_, length_ * size_per_head_,
	query_buf, BType_, size_per_head_, length_ * size_per_head_,
	&beta,
	qk_buf, CType_, length_, length_ * length_,
	batch_size_ * head_num_,
	computeType_,
	static_cast<cublasGemmAlgo_t>(cublasAlgo_)));
    //add bias
    postprocess(qk_buf_, bias_buf_, batch_size_ * length_, hidden_units_);
    //compute softmax
    dim3 grid(batch_size_ * head_num_);
    dim3 block(1024);
    softmax_kernel<DataType_> <<<grid, block, 0, param.stream>>>(qk_buf_, softmax_buf_, batch_size_, head_num_, length_);

	
     //compute softmax*v
    beta = (DataType_)0.0f;
    check_cuda_error(cublasGemmStridedBatchedEx(param.cublas_handle,
                                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                                  size_per_head_, length_, length_,
                                                  &alpha,
                                                  value_buf, AType_, size_per_head_, length_ * size_per_head_,
                                                  softmax_buf_, BType_, length_, length_ * length_,
                                                  &beta,
                                                  qv_buf, CType_, size_per_head_, length_ * size_per_head_,
                                                  batch_size_ * head_num_,
                                                  computeType_,
                                                  static_cast<cublasGemmAlgo_t>(cublasAlgo_)));
    //compute softmax * position_value + softmax * v
    beta = 1.0f;
    batchCount = batch_size_ * length_ * head_num_;
    for(int i=0; i<batchCount; ++i)
    {
    	h_a_array[i] = softmax_buf_ + i * length_;
    	h_b_array[i] = position_value + length_ * (hidden_units_ / head_num_) * (i / head_num_ % length_);
    	h_c_array[i] = qv_buf + i * (hidden_units_ / head_num_);
    }

    cudaMalloc((void**)&a_array, batchCount * sizeof(float *));
    cudaMalloc((void**)&b_array, batchCount * sizeof(float *));
    cudaMalloc((void**)&c_array, batchCount * sizeof(float *));
    cudaMemcpy(a_array, thrust::raw_pointer_cast(&h_a_array[0]), batchCount * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(b_array, thrust::raw_pointer_cast(&h_b_array[0]), batchCount * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(c_array, thrust::raw_pointer_cast(&h_c_array[0]), batchCount * sizeof(float *), cudaMemcpyHostToDevice);

    
    lda = length_;
    ldb = hidden_units_ / head_num_;
    ldc = length_;
    
    check_cuda_error(cublasGemmBatchedEx(param.cublas_handle,
		        CUBLAS_OP_N,
			CUBLAS_OP_N,
		        hidden_units_/head_num_,
			1,
			length_,
			&alpha,
			b_array, BType_, ldb,
			a_array, AType_, lda,
			&beta,
			c_array, CType_, ldc,
			batchCount,
			computeType_,
			static_cast<cublasGemmAlgo_t>(cublasAlgo_)));
    //last dense
   beta = 0.0f;
   check_cuda_error(cublasGemmEx(param.cublas_handle, 
                        	  CUBLAS_OP_N, CUBLAS_OP_N, 
                        	  n, m, k, 
                        	  &alpha, 
                        	  param.self_attention.attention_output_weight, AType_, n, 
                        	  qv_buf, BType_, k, 
                        	  &beta, 
                        	  input_tensor, CType_, n, 
                        	  computeType_, 
                        	  static_cast<cublasGemmAlgo_t>(cublasAlgo_)));

}


template<OperationType OpType_>
void OpenEncoder<OpType_>::ffn(const DataType_* input, DataType_* ffn_inner, DataType_* output,
                               const int m, const int inner_size, const int n, ParaMeter_ param) 
{
  int m1 = m, k1 = n, n1 = inner_size;
  DataType_ alpha = (DataType_)1.0f;
  DataType_ beta = (DataType_)0.0f;
  cublasGemmEx(param.cublas_handle,CUBLAS_OP_N, CUBLAS_OP_N, 
               n1, m1, k1, 
               &alpha, 
               param.ffn.intermediate_weight.kernel, AType_, n1, 
               input, BType_, k1, 
               &beta, 
               ffn_inner, CType_, n1, 
               computeType_, 
               static_cast<cublasGemmAlgo_t>(cublasAlgo_));

  dim3 grid(m1);
  dim3 block(n1 / 4);
  add_bias_relu<DataType_><<<grid, block, 0, param.stream>>>(ffn_inner, param.ffn.intermediate_weight.bias, m1, n1);
  int m2 = m, n2 = n, k2 = inner_size;


  check_cuda_error(cublasGemmEx(param.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n2, m2, k2, &alpha,
                   param.ffn.output_weight.kernel, AType_, n2, ffn_inner, BType_, k2,
                   &beta, output, CType_, n2,computeType_, static_cast<cublasGemmAlgo_t>(cublasAlgo_)));


  grid = m;
  block = n;
  add_bias_kernel<DataType_><<<grid, block, 0, param.stream>>>(output, param.ffn.output_weight.bias, m, n);
}


template<OperationType OpType_>
void OpenEncoder<OpType_>::postprocess(DataType_* output, const DataType_* input, const int m, const int n)
{
  dim3 grid(m);
  dim3 block(n);
  add_input_kernel<DataType_><<<grid, block, 0, param_[0].stream>>>(output, input,  m, n);
}

template<OperationType OpType_>
void OpenEncoder<OpType_>::set_zero(DataType_* input, const int batch, const int length, const int hidden_units, int* mask)
{
  dim3 grid(batch*length);
  dim3 block(hidden_units);
  set_zero_kernel<<<grid, block, 0, param_[0].stream>>>(input, mask, hidden_units);
}

template<OperationType OpType_>
void OpenEncoder<OpType_>::embedding_relative_position(DataType_* position_key, DataType_* position_value, int* mat_buf, const int length, ParaMeter_ param) 
{
   dim3 grid(length * length);
   dim3 block(64);
   position_embedding_kernel<<<grid, block, 0, param.stream>>>(position_key, position_value, param.self_attention.position_key, param.self_attention.position_value, mat_buf);
}

template<OperationType OpType_>
void OpenEncoder<OpType_>::build_position_mat(int* mat_buf, const int max_length, const int length) 
{
    int max = 2 * max_length;
    int* mat = new int[length * length];
    //get position and encode
    for (int i = 0; i < length * length; i++ )
    {
	int tmp = i % length - (i / length) + max_length;
	mat[i] = tmp > max? max:tmp;
	if (tmp < 0) mat[i] = 0;
    }
    cudaMemcpy(mat_buf, mat, length * length * sizeof(int), cudaMemcpyHostToDevice);
    delete []mat;
}

template<OperationType OpType_>
void OpenEncoder<OpType_>::build_bias(DataType_* bias_buf, int batch, int length,  int* mask) 
{
    dim3 grid(batch * length);
    dim3 block(1);
    build_bias_kernel<<<grid, block, 0, param_[0].stream>>>(bias_buf, mask);
}

template <OperationType OpType_>
void OpenEncoder<OpType_>::forward(int *input, int *mask,  int *language)
{
        initialize(buf_);
	embedding_lookup(p_table_embedding_, p_table_language_, input, input_tensor_buf_, batch_size_, length_, mask, language);
	build_bias(bias_buf_, batch_size_, length_, mask);
        build_position_mat(mat_buf_, 20, length_);
	for(int i=0; i<layer_num_; i++)
	{
		try
		{
			//preprocess
                       
			preprocess(input_tensor_buf_, output_tensor_buf_, length_, batch_size_, hidden_units_, param_[i]);
			//get relative_position
			embedding_relative_position(position_key_buf_, position_value_buf_, mat_buf_, length_, param_[i]);
			//attention
			encoder_attention(output_tensor_buf_, q_tensor_buf_, k_tensor_buf_, v_tensor_buf_, position_key_buf_, position_value_buf_, qk_buf_, qv_buf_, param_[i]);
			//postprocess
			postprocess(input_tensor_buf_, output_tensor_buf_, batch_size_ * length_, hidden_units_);
                        //preprocess
			preprocess(input_tensor_buf_, output_tensor_buf_, length_, batch_size_, hidden_units_, param_[i]);
			//ffn
			ffn(output_tensor_buf_, inner_tensor_buf_, ffn_out_buf_, batch_size_*length_ , inner_size_, hidden_units_, param_[i]);
			//set_sero
			set_zero(ffn_out_buf_, batch_size_, length_, hidden_units_, mask);
			//postprocess
			postprocess(input_tensor_buf_, ffn_out_buf_,  batch_size_ * length_, hidden_units_);
		}
		catch (std::runtime_error &error)
		{
			throw error;
		}

	}
	lastprocess(input_tensor_buf_, encoder_result_buf_, length_, batch_size_, hidden_units_, p_gamma_, p_beta_);
}


template  OpenEncoder<OperationType::FP32>::OpenEncoder(const IAllocator &allocator, const int& batch_size, const int& length,
              const int& head_num, const int& size_per_head, const int& hidden_units,
              const int& memory_hidden_units, const int& layer_num, ParaMeter_* param,
              DataType_ * p_table_embedding, DataType_ * p_table_language,
              DataType_ * p_gamma, DataType_ * p_beta);

template  void OpenEncoder<OperationType::FP32>::initialize(DataType_* buf);

template  void OpenEncoder<OperationType::FP32>::forward(int *input, int *mask,  int *language);

template  void OpenEncoder<OperationType::FP32>::build_bias(DataType_* bias_buf, int batch, int length,  int* mask);

template  void OpenEncoder<OperationType::FP32>::build_position_mat(int* mat_buf, const int max_length, const int length);

template  void OpenEncoder<OperationType::FP32>::postprocess(DataType_* output, const DataType_* input, const int m, const int n);

template  void OpenEncoder<OperationType::FP32>::preprocess(const DataType_* input, DataType_* output, int length, int batch_size,
                  int hidden_units, ParaMeter_ param);

template  void OpenEncoder<OperationType::FP32>::embedding_relative_position(DataType_* position_key, DataType_* position_value,
                              int* mat_buf, const int length, ParaMeter_ param);

template  void OpenEncoder<OperationType::FP32>::embedding_lookup(const DataType_* embedding_table, const DataType_* embedding_language,
                   const int* input_ids, DataType_* out_tensor, const int batch_size,
                   const int length, const int* mask, const int* language);

template  void OpenEncoder<OperationType::FP32>::encoder_attention(DataType_* input_tensor, DataType_ * query_buf, DataType_* key_buf,
                    DataType_* value_buf, DataType_* position_key, DataType_* position_value,
                    DataType_* qk_buf, DataType_ * qv_buf, ParaMeter_ param);

template  void OpenEncoder<OperationType::FP32>::ffn(const DataType_* input, DataType_* ffn_inner, DataType_* output,
                  const int m, const int inner_size, const int n, ParaMeter_ param);

template  void OpenEncoder<OperationType::FP32>::set_zero(DataType_* input, const int batch, const int length, const int hidden_units, int* mask);

template  void OpenEncoder<OperationType::FP32>::lastprocess(const DataType_* input, DataType_* output, int length, int batch_size, int hidden_units,
              const DataType_* gamma, const DataType_* beta);


}
