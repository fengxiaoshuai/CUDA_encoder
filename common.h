/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>


template<typename T>
struct DenseWeight
{
    const T* kernel;
    const T* bias;
};

template<typename T>
struct LayerNormWeight
{
    const T* gamma;
    const T* beta;
};

template<typename T>
struct AttentionWeight
{
    const T* query_weight;
    const T* key_weight;
    const T* value_weight;
    const T* attention_output_weight;
    const T* position_key;
    const T* position_value;
};

template<typename T>
struct FFNWeight
{
    DenseWeight<T> intermediate_weight;
    DenseWeight<T> output_weight;
};

namespace fastertransformer{

enum class OperationType{FP32, FP16};
enum class AllocatorType{CUDA, TF};

static const char *_cudaGetErrorEnum(cudaError_t error) 
{
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error) 
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}


template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + \
        (_cudaGetErrorEnum(result)) + " " + file +  \
        ":" + std::to_string(line) + " \n");\
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)


template<OperationType OpType_>
class TransformerTraits;

template<>
class TransformerTraits<OperationType::FP32>
{
  public:
    typedef float DataType;
    static const OperationType OpType = OperationType::FP32;
    static cudaDataType_t const computeType = CUDA_R_32F;
    static cudaDataType_t const AType = CUDA_R_32F;
    static cudaDataType_t const BType = CUDA_R_32F;
    static cudaDataType_t const CType = CUDA_R_32F;
};

template<>
class TransformerTraits<OperationType::FP16>
{
  public:
    typedef half DataType;
    static const OperationType OpType = OperationType::FP16;
    static cudaDataType_t const computeType = CUDA_R_16F;
    static cudaDataType_t const AType = CUDA_R_16F;
    static cudaDataType_t const BType = CUDA_R_16F;
    static cudaDataType_t const CType = CUDA_R_16F;
};

}
