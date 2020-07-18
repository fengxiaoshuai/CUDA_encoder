nvcc -o test  main.cpp  encoder.cu  -lcublas -lcudart -arch=sm_75 --std=c++11
