#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;

#define GRID_SIZE 16
#define BLOCK_SIZE 16
#define NUM_LISTS 16
unsigned long const NUM_ELEMENT = (1 << 10);

template <class T>
void c_swap(T &x, T &y) {
  T tmp1 = x;
  x = y;
  y = tmp1;
}

__device__ void radix_sort(unsigned long *const sort_tmp,
                           unsigned long *const sort_tmp_1,
                           unsigned long *const sort_tmp11) {
  for (unsigned long bit_mask = 1; bit_mask > 0; bit_mask <<= 1) {
    unsigned int base_cnt_0 = 0;
    unsigned int base_cnt_1 = 0;

    for (int i = 0; i < NUM_ELEMENT; i += NUM_LISTS) {
      if (tid + i < NUM_ELEMENT) {
        if (sort_tmp[i + tid] & bit_mask) {
          sort_tmp_1[base_cnt_1 + tid] = sort_tmp[i + tid];
          base_cnt_1 += NUM_LISTS;
        } else {
          sort_tmp[base_cnt_0 + tid] = sort_tmp[i + tid];
          base_cnt_0 += NUM_LISTS;
        }
      }
    }

    for (unsigned long i = 0; i < base_cnt_1; i += NUM_LISTS) {
      if (tid + i < NUM_ELEMENT) {
        sort_tmp[base_cnt_0 + i + tid] = sort_tmp_1[i + tid];
      }
    }
    __syncthreads();
  }
}


__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
if (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
if (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16];
if (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8];
if (BLOCK_SIZE >= 8) sdata[tid] += sdata[tid + 4];
if (BLOCK_SIZE >= 4) sdata[tid] += sdata[tid + 2];
if (BLOCK_SIZE >= 2) sdata[tid] += sdata[tid + 1];
}
__global__ void sort_block(int *g_idata, int *g_odata, unsigned int n) {
extern __shared__ int sdata[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + tid;
unsigned int gridSize = BLOCK_SIZE*2*gridDim.x;
sdata[tid] = 0;
while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+BLOCK_SIZE]; i += gridSize; }
__syncthreads();
if (BLOCK_SIZE >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
if (BLOCK_SIZE >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
if (BLOCK_SIZE >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
if (tid < 32) warpReduce(sdata, tid);
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__device__ int search_index(unsigned long *const array_tmp, unsigned long val) {
  int left = 0;
  int right = NUM_ELEMENT - 1;
  while (left <= right) {
    int middle = (right + left) / 2;
    if (array_tmp[middle] > val) {
      right = middle - 1;

    } else if (array_tmp[middle] < val) {
      left = middle + 1;
    } else {
      return middle;
    }
  }
  return -1;
  //二分法找元素index
}
template <typename S>
__global__ void sort_index(S *sortarray, unsigned long *const array_tmp,
                           unsigned long *const data) {
  const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int tid = ix + iy * (gridDim.x * blockDim.x);
  for (int i = 0; i < NUM_ELEMENT; i += NUM_LISTS) {
    if (tid + i < NUM_ELEMENT) {
      data[tid + i] = search_index(array_tmp, sortarray[i + tid].key);
    }
  }
  __syncthreads();
  for (int i = 0; i < NUM_ELEMENT; i += NUM_LISTS) {
    if (tid + i < NUM_ELEMENT) {
      array_tmp[data[tid + i]] = tid + i;
    }
  }
  __syncthreads();
}

template <typename S>
__global__ void sort_struct(unsigned long *const array_tmp, S *sortarray,
                            S *struct_tmp) {
  const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int tid = ix + iy * (gridDim.x * blockDim.x);
  for (int i = 0; i < NUM_ELEMENT; i += NUM_LISTS) {
    if (tid + i < NUM_ELEMENT) {
      struct_tmp[tid + i] = sortarray[array_tmp[tid + i]];
    }
  }
  __syncthreads();

  for (int i = 0; i < NUM_ELEMENT; i += NUM_LISTS) {
    if (tid + i < NUM_ELEMENT) {
      sortarray[tid + i] = struct_tmp[tid + i];
    }
  }
  __syncthreads();
}

template <typename S>
__device__ int search(S *nums, unsigned long val) {
  for (int i = 0; i < NUM_ELEMENT; i++) {
    if (nums[i].key == val) {
      return i;
    }
  }
  return -1;
}
typedef struct SORTSTRUCT {
  unsigned long key;
} sorta;

sorta sortarray[NUM_ELEMENT];  //定义为全局变量避免堆栈溢出

int main(void) {
  for (unsigned long i = 0; i < NUM_ELEMENT; i++) {
    sortarray[i].key = i;
    // sortarray[i].key = i%35;//key值相等的情况
  }

  for (int i = 0; i < NUM_ELEMENT; i++) {
    c_swap(sortarray[rand() % 7].key, sortarray[i].key);
  }
  sorta *gpu_sortarray;
  sorta *struct_tmp;
  unsigned long *array_tmp;
  unsigned long *array_tmp_2;
  unsigned long *sort_tmp;
  unsigned long *min_array;
  unsigned long *min_value;
  cudaMalloc((sorta **)&gpu_sortarray, sizeof(sorta) * NUM_ELEMENT);
  cudaMalloc((void **)&array_tmp, sizeof(unsigned long) * NUM_ELEMENT);
  cudaMalloc((void **)&array_tmp_2, sizeof(unsigned long) * NUM_ELEMENT);
  cudaMalloc((void **)&sort_tmp, sizeof(unsigned long) * NUM_LISTS);
  cudaMalloc((void **)&min_array, sizeof(unsigned long) * GRID_SIZE);
  cudaMalloc((sorta **)&struct_tmp, sizeof(sorta) * NUM_ELEMENT);

  radix_sort<<<GRID_SIZE, BLOCK_SIZE>>>(gpu_sortarray, array_tmp, sort_tmp);

  for (int i = 0; i < NUM_ELEMENT; i++) {
    min_value = 0xFFFFFFF;
    merge_block<<<GRID_SIZE, BLOCK_SIZE>>>(sort_tmp, min_array);
    merge_final<<<1, BLOCK_SIZE>>>(min_array, min_value);
    for (int j = 0; j < NUM_LISTS; j++) {
      if (self_data[j] == min_value) {
        sortarray_tmp[i] = min_value;
        index[j] = index[j] + 1;
      }
      if (j + index[j] * NUM_LISTS < NUM_ELEMENT) {
        sort_tmp[j] = array_tmp[j + index[j] * NUM_LISTS];
      } else {
        sort_tmp[j] = 0xFFFFFFFF;
      }
    }
  }
  sort_index<<<GRID_SIZE, BLOCK_SIZE>>>(gpu_sortarray, array_tmp, array_tmp_2);
  sort_struct<<<GRID_SIZE, BLOCK_SIZE>>>(array_tmp, gpu_sortarray, struct_tmp);
  ///////////////////////////////////////////
  cudaMemcpy(gpu_sortarray, sortarray, sizeof(sorta) * NUM_ELEMENT,
             cudaMemcpyHostToDevice);


  cudaEvent_t start, stop;  //定义事件
  cudaEventCreate(&start);  //起始时间
  cudaEventCreate(&stop);   //结束时间

  cudaEventRecord(start, 0);  //记录起始时间
  



  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);  //执行完代码，记录结束时间

  cudaEventSynchronize(stop);
  cudaMemcpy(sortarray, array_tmp, sizeof(sorta) * NUM_ELEMENT,
             cudaMemcpyDeviceToHost);
  cudaError_t error = cudaGetLastError();
  cudaFree(gpu_srcData);
  cudaFree(array_tmp);
  cudaFree(gpu_sortarray);
  cudaFree(struct_tmp);

  printf("CUDA error: %s\n", cudaGetErrorString(error));

  int result = 0;
  for (int i = 0; i < NUM_ELEMENT - 1; i++) {
    if (sortarray[i].key > sortarray[i + 1].key) {
      result++;
    }
    // printf("%ld\n",sortarray[i].key);
  }

  for (int i = 0; i < NUM_ELEMENT; i++) {
    printf("%ld\n", sortarray[i].key);
  }

  printf("%ld\n", NUM_ELEMENT);
  printf("%d\n", result);
  if (result == 0) {
    printf("result is true.\n");
  } else {
    printf("result is false.\n");
  }
  float elapsedTime;  //计算总耗时，单位ms
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("%f\n", elapsedTime);
}
