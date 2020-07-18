#ifndef _ENCODER_H_
#define _ENCODER_H_

#include "allocator.h"
#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fastertransformer
{

template <typename T>
class EncoderInitParam
{
public:
  LayerNormWeight<T> self_layernorm;
  AttentionWeight<T> self_attention;
  LayerNormWeight<T> ffn_layernorm;
  FFNWeight<T> ffn;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

template <OperationType OpType_>
class EncoderTransformerTraits;

template <>
class EncoderTransformerTraits<OperationType::FP32> : public TransformerTraits<OperationType::FP32>{};

template <>
class EncoderTransformerTraits<OperationType::FP16> : public TransformerTraits<OperationType::FP16>{};


template <OperationType OpType_>
class OpenEncoder
{
private:
  typedef EncoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  typedef EncoderInitParam<DataType_> ParaMeter_;
  

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  int cublasAlgo_;


  const IAllocator &allocator_;
  ParaMeter_ *param_;
  DataType_ *p_table_embedding_; 
  DataType_ *p_table_language_; 
  DataType_ *p_gamma_; 
  DataType_ *p_beta_; 
 
  int batch_size_;//未知
  int length_;//未知
  int head_num_;//16
  int layer_num_;//6
  int size_per_head_;//64
  int hidden_units_;//1024
  int inner_size_;//4096
  
  int       *mat_buf_; 
  DataType_ *buf_;
  DataType_ *input_tensor_buf_; 
  DataType_ *output_tensor_buf_; 
  DataType_ *q_tensor_buf_; 
  DataType_ *k_tensor_buf_; 
  DataType_ *v_tensor_buf_; 
  DataType_ *qk_buf_; 
  DataType_ *qv_buf_; 
  DataType_ *bias_buf_; 
  DataType_ *softmax_buf_;
  DataType_ *position_value_buf_;
  DataType_ *position_key_buf_;
  DataType_ *inner_tensor_buf_;
  DataType_ *ffn_out_buf_;
  DataType_ *encoder_result_buf_;
  
public:
  OpenEncoder(const IAllocator &allocator, const int& batch_size, const int& length,
              const int& head_num, const int& size_per_head, const int& hidden_units,
	      const int& memory_hidden_units, const int& layer_num, ParaMeter_* param,
              DataType_ * p_table_embedding, DataType_ * p_table_language, 
	      DataType_ * p_gamma, DataType_ * p_beta);

  void initialize(DataType_* buf);

  void forward(int *input, int *mask,  int *language);

  void build_bias(DataType_* bias_buf, int batch, int length,  int* mask);

  void build_position_mat(int* mat_buf, const int max_length, const int length);

  void postprocess(DataType_* output, const DataType_* input, const int m, const int n);

  void preprocess(const DataType_* input, DataType_* output, int length, int batch_size,
                  int hidden_units, ParaMeter_ param);

  void embedding_relative_position(DataType_* position_key, DataType_* position_value,
                              int* mat_buf, const int length, ParaMeter_ param);

  void embedding_lookup(const DataType_* embedding_table, const DataType_* embedding_language, 
                   const int* input_ids, DataType_* out_tensor, const int batch_size,
                   const int length, const int* mask, const int* language);

  void encoder_attention(DataType_* input_tensor, DataType_ * query_buf, DataType_* key_buf,
                    DataType_* value_buf, DataType_* position_key, DataType_* position_value,
                    DataType_* qk_buf, DataType_ * qv_buf, ParaMeter_ param);

  void ffn(const DataType_* input, DataType_* ffn_inner, DataType_* output,
      const int m, const int inner_size, const int n, ParaMeter_ param);

  void set_zero(DataType_* input, const int batch, const int length, const int hidden_units, int* mask);

  void lastprocess(const DataType_* input, DataType_* output, int length, int batch_size, int hidden_units,
              const DataType_* gamma, const DataType_* beta);

  ~OpenEncoder(){};


};

}
#endif
