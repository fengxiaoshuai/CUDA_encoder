#include "encoder.h"
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>

using namespace fastertransformer;
using namespace std;


void load(vector<float>& weight, string dir)
{
        ifstream input(dir);

        if (input.fail())
        {
                cout << "File does not exist" << endl;
                cout << "Exit program" << endl;
                return;
        }
        float num=0.0;
        while (input>>num)  // 当没有读到文件结尾
        {
                weight.push_back(num);
                //cout << num << endl;
        }
        input.close();
}

void load_layer_weight(vector<vector<float>>& layer_weight, int num)
{
        cout << "start read layer " << num << " weight" << endl;
        vector<float> layer_self_scale;//0
        vector<float> layer_self_bias;//1
        vector<float> layer_self_q;//2
        vector<float> layer_self_k;//3
        vector<float> layer_self_v;//4
        vector<float> layer_self_last;//5

        vector<float> layer_ffn_scale;//6
        vector<float> layer_ffn_bias;//7
        vector<float> layer_ffn_first_weight;//8
        vector<float> layer_ffn_first_bias;//9
        vector<float> layer_ffn_second_weight;//10
        vector<float> layer_ffn_second_bias;//11

        vector<float> layer_self_position_key;//12
        vector<float> layer_self_position_value;//13


        cout << "...:load self attention weight" << endl;
        string name = "./weight/layer_" + to_string(num) ;
        load(layer_self_scale, name + "_self_scale.txt");
        load(layer_self_bias, name + "_self_bias.txt");
        load(layer_self_q, name + "_self_q.txt");
        load(layer_self_k, name + "_self_k.txt");
        load(layer_self_v, name + "_self_v.txt");
        load(layer_self_last, name + "_self_last.txt");
        load(layer_self_position_key, name + "_self_position_key.txt");
        load(layer_self_position_value, name + "_self_position_value.txt");

        cout << "...:load read fnn weight" << endl;
        load(layer_ffn_scale, name + "_ffn_scale.txt");
        load(layer_ffn_bias, name + "_ffn_bias.txt");
        load(layer_ffn_first_weight, name + "_ffn_first_weight.txt");
        load(layer_ffn_first_bias, name + "_ffn_first_bias.txt");
        load(layer_ffn_second_weight, name + "_ffn_second_weight.txt");
        load(layer_ffn_second_bias, name + "_ffn_second_bias.txt");


        layer_weight.push_back(layer_self_scale);
        layer_weight.push_back(layer_self_bias);
        layer_weight.push_back(layer_self_q);
        layer_weight.push_back(layer_self_k);
        layer_weight.push_back(layer_self_v);
        layer_weight.push_back(layer_self_last);

        layer_weight.push_back(layer_ffn_scale);
        layer_weight.push_back(layer_ffn_bias);
        layer_weight.push_back(layer_ffn_first_weight);
        layer_weight.push_back(layer_ffn_first_bias);
        layer_weight.push_back(layer_ffn_second_weight);
        layer_weight.push_back(layer_ffn_second_bias);

        layer_weight.push_back(layer_self_position_key);
        layer_weight.push_back(layer_self_position_value);

        cout << "...:end layer " << num << " weight" << endl;
}


template<typename T>
void BuildBias(const int& batch_size, const int& length,  int* mask, T* bias)
{
        for (int i = 0; i < batch_size*length; i++)
        {
                bias[i] *= (1-mask[i]);
        }
}

template<typename T>
void device_malloc(T** ptr, int size, T* h_ptr)
{
  check_cuda_error(cudaMalloc((void**)ptr, size));
  check_cuda_error(cudaMemcpy(*ptr, h_ptr, size, cudaMemcpyHostToDevice));
}

template<typename T>
void encoding_sample(const int batch_size,
                     const int head_num,
                     const int size_per_head,
                     const int vocab_size,
                     const int length,
                     const int encoder_layers,
                     const int hidden_units,
                     const int language_num,
                     int* language_id,
                     int* mask,
		     int* input)
{
  const int inner_size = 4096;
  const int max_position = 20;


  vector<float> weight_embedding;
  load(weight_embedding, "./weight/embedding.txt");
  vector<float> language_embedding;
  load(language_embedding, "./weight/language_embedding.txt");
  vector<float> weight_scale;
  load(weight_scale, "./weight/scale.txt");
  vector<float> weight_bias;
  load(weight_bias, "./weight/bias.txt");

  vector<vector<vector<float>>> weight(encoder_layers);
  for(int i = 0; i<encoder_layers; i++)
  {
        load_layer_weight(weight[i], i);
  }


  cublasHandle_t cublasHandle;
  check_cuda_error(cublasCreate(&cublasHandle));

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  check_cuda_error(cublasSetStream(cublasHandle, stream));

  fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
  EncoderInitParam<T> *param = new EncoderInitParam<T>[encoder_layers];
  /*
  T h_bias[batch_size * length] = {-1e9};
  for(int i=0; i<batch_size * length; i++)
  {
    h_bias[i] = -1e9;
  }
  BuildBias(batch_size, length, mask, h_bias);
  */
  cout << "start malloc for GPU" << endl;
  for(int i = 0; i < encoder_layers; i++)
  {
    param[i].stream = stream;
    param[i].cublas_handle = cublasHandle;

    T *d_self_Q_kernel, *d_self_K_kernel, *d_self_V_kernel, *d_self_output_kernel, *d_self_gamma, *d_self_beta;
    T *d_self_position_key, *d_self_position_value;
    T *d_ffn_kernel1, *d_ffn_bias1, *d_ffn_kernel2, *d_ffn_bias2, *d_ffn_gamma, *d_ffn_beta;

    device_malloc(&d_self_gamma, sizeof(T) * hidden_units, weight[i][0].data());
    device_malloc(&d_self_beta, sizeof(T) * hidden_units, weight[i][1].data());
    device_malloc(&d_self_Q_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][2].data());
    device_malloc(&d_self_K_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][3].data());
    device_malloc(&d_self_V_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][4].data());
    device_malloc(&d_self_output_kernel, sizeof(T) * hidden_units * hidden_units, weight[i][5].data());
    device_malloc(&d_self_position_key, sizeof(T) * (max_position*2+1) * size_per_head, weight[i][12].data());
    device_malloc(&d_self_position_value, sizeof(T) * (max_position*2+1) * size_per_head, weight[i][13].data());

    device_malloc(&d_ffn_gamma, sizeof(T) * hidden_units, weight[i][6].data());
    device_malloc(&d_ffn_beta, sizeof(T) * hidden_units, weight[i][7].data());
    device_malloc(&d_ffn_kernel1, sizeof(T) * inner_size * hidden_units, weight[i][8].data());
    device_malloc(&d_ffn_bias1, sizeof(T) * inner_size, weight[i][9].data());
    device_malloc(&d_ffn_kernel2, sizeof(T) * inner_size * hidden_units, weight[i][10].data());
    device_malloc(&d_ffn_bias2, sizeof(T) * hidden_units, weight[i][11].data());


    param[i].self_layernorm.gamma = d_self_gamma;
    param[i].self_layernorm.beta = d_self_beta;
    param[i].self_attention.query_weight = d_self_Q_kernel;
    param[i].self_attention.key_weight = d_self_K_kernel;
    param[i].self_attention.value_weight = d_self_V_kernel;
    param[i].self_attention.attention_output_weight = d_self_output_kernel;
    param[i].self_attention.position_key = d_self_position_key;
    param[i].self_attention.position_value = d_self_position_value;


    param[i].ffn_layernorm.gamma = d_ffn_gamma;
    param[i].ffn_layernorm.beta = d_ffn_beta;
    param[i].ffn.intermediate_weight.kernel = d_ffn_kernel1;
    param[i].ffn.intermediate_weight.bias = d_ffn_bias1;
    param[i].ffn.output_weight.kernel = d_ffn_kernel2;
    param[i].ffn.output_weight.bias = d_ffn_bias2;
  }

  T *d_table_language;
  T *d_table_embedding;
  T *d_gamma;
  T *d_beta;

  device_malloc(&d_table_language, sizeof(T) * language_num  *hidden_units , language_embedding.data());
  device_malloc(&d_table_embedding, sizeof(T) * vocab_size * hidden_units , weight_embedding.data());
  device_malloc(&d_gamma, sizeof(T) * hidden_units, weight_scale.data());
  device_malloc(&d_beta, sizeof(T) * hidden_units, weight_bias.data());


  const fastertransformer::OperationType type = sizeof(T) == sizeof(float) ? OperationType::FP32 : OperationType::FP16;

  cout << "end malloc for GPU" << endl;
 
  OpenEncoder<type> *encoder = new OpenEncoder<type>(allocator, batch_size, length, head_num,
						     size_per_head, hidden_units, inner_size,
                                                     encoder_layers, param, d_table_embedding,
                                                     d_table_language, d_gamma, d_beta);

  int *d_input;
  int *d_mask;
  int *d_language;
  device_malloc(&d_input, sizeof(int) * batch_size * length, input);
  device_malloc(&d_mask, sizeof(int) * batch_size * length, mask);
  device_malloc(&d_language, sizeof(int) * batch_size, language_id);

  int ite = 5;
  for(int i = 0; i < ite; ++i)
  {
	encoder->forward(d_input, d_mask, d_language);
  }
  cudaDeviceSynchronize();

  struct timeval start, end;
  gettimeofday(&start, NULL);
  for(int i = 0; i < 10; ++i)
  {
	encoder->forward(d_input, d_mask, d_language);
  }
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);

  printf("time: %.2f ms \n",((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / 10);
  delete encoder;
  return ;
}



int main(int argc, char* argv[])
{

  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);


  const int batch_size = 2;
  const int head_num = 16;
  const int size_per_head = 64;
  const int vocab_size = 32768;
  const int length = 8;
  const int encoder_layer = 6;
  const int hidden_unit = 1024;
  const int language_num = 4;
  const int encoder_layers = 6;
  int language_id[batch_size] = {1,1};
  int mask[batch_size * length] = {1,1,1,1,1,0,0,0, 1,1,1,1,1,1,0,0};
  int input[16] = {115, 29, 112, 18, 17036, 0, 0, 0,   177, 6716, 7667,  9643, 8, 124, 0, 0};

  encoding_sample<float>(batch_size, head_num, size_per_head, vocab_size, length, encoder_layers, hidden_unit, language_num, language_id, mask, input);

  return 0;
}
           
