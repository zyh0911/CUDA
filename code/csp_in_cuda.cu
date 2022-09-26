// csp in cuda new
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

#define NUM_LISTS 16
#define NUM_GRIDS 1
unsigned long const NUM_ELEMENT = (1 << 6);

template <class T>
void c_swap(T &x, T &y) {
  T tmp1 = x;
  x = y;
  y = tmp1;
}

template <typename S>
__device__ void copy_index(S *sortarray, unsigned long *const data,
                           const unsigned int tid) {
  for (int i = 0; i < NUM_ELEMENT; i += NUM_LISTS) {
    data[tid + i] = sortarray[tid + i].key;
  }
  __syncthreads();
}

__device__ void radix_sort(unsigned long *const sort_tmp,
                           unsigned long *const sort_tmp_1,
                           const unsigned int tid) {
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
//这里有问题，block之间无法同步，导致信息传输有问题
__device__ void merge(unsigned long *const data, unsigned long *const array_tmp,
                      const unsigned int tid,unsigned long *const self_data,unsigned long min_data
                      ) {
  //__shared__ 
  unsigned int index[NUM_LISTS];
  //__shared__
  //unsigned int min_data;
  //__shared__
  //unsigned int min_tid;
  index[tid] = 0;
  __threadfence();

  for (int i = 0; i < NUM_ELEMENT; i++) {
    //__shared__ unsigned int self_data[NUM_LISTS];
    self_data[tid] = 0xFFFFFFFF;
    min_data = 0xFFFFFFFF;
    
    

    if (tid + index[tid] * NUM_LISTS < NUM_ELEMENT) {
      self_data[tid] = data[tid + index[tid] * NUM_LISTS];
      __threadfence();
    } else {
      self_data[tid] = 0xFFFFFFFF;
      __threadfence();
    }
    
    __threadfence();
    
    for (int j = 0; j < NUM_LISTS; j++) {
      min_data = min(min_data, self_data[j]);
    }
    //atomicMin(&min_data,self_data[tid]);
    __syncthreads();
    //__threadfence();
    //printf("%ld\n",min_data);
    for(int j=0;j<NUM_LISTS;j++)
    {
    if (self_data[j] == min_data) 
    /*{
      min_tid = min(min_tid, tid);
    }
    if (tid == min_tid) */
    {
      array_tmp[i] = min_data;
      index[j] = index[j] + 1;
    }

    }
    __threadfence();
  }
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
__device__ void sort_index(S *sortarray, unsigned long *const array_tmp,
                           unsigned long *const data, const unsigned int tid) {
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
__device__ void sort_struct(unsigned long *const array_tmp, S *sortarray,
                            S *struct_tmp, const unsigned int tid) {
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

__global__ void cspincuda(unsigned long *const data,
                          unsigned long *const array_tmp, sorta *sortarray,
                          sorta *struct_tmp,unsigned long *const self_data,
                          unsigned long min_data) {
  const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int tid = ix + iy * (gridDim.x * blockDim.x);
  copy_index(sortarray, data, tid);  // step1:copy index
  radix_sort(data, array_tmp, tid);
  merge(data, array_tmp, tid,self_data,min_data);
  //sort_index(sortarray, array_tmp, data, tid);         // step2:sort_by_key
  //sort_struct(array_tmp, sortarray, struct_tmp, tid);  // step3:sort array
}

sorta sortarray[NUM_ELEMENT];  //定义为全局变量避免堆栈溢出

int main(void) {
  for (unsigned long i = 0; i < NUM_ELEMENT; i++) {
    sortarray[i].key = i;
    // sortarray[i].key = i%35;//key值相等的情况
  }

  for (int i = 0; i < NUM_ELEMENT; i++) {
    c_swap(sortarray[rand() % 7].key, sortarray[i].key);
  }

  unsigned long *gpu_srcData;
  unsigned long *array_tmp;
  sorta *gpu_sortarray;
  sorta *struct_tmp;
  unsigned long *self_data;
  unsigned long min_data;
  cudaMalloc((void **)&gpu_srcData, sizeof(unsigned long) * NUM_ELEMENT);
  cudaMalloc((void **)&array_tmp, sizeof(unsigned long) * NUM_ELEMENT);
  cudaMalloc((void **)&self_data, sizeof(unsigned long) * NUM_LISTS);
  cudaMalloc((void **)&min_data, sizeof(unsigned long));
  cudaMalloc((sorta **)&gpu_sortarray, sizeof(sorta) * NUM_ELEMENT);
  cudaMalloc((sorta **)&struct_tmp, sizeof(sorta) * NUM_ELEMENT);

  cudaMemcpy(gpu_sortarray, sortarray, sizeof(sorta) * NUM_ELEMENT,
             cudaMemcpyHostToDevice);

  // cudaError_t error = cudaGetLastError();
  dim3 grid(2);
  dim3 block(8);

  cudaEvent_t start, stop;  //定义事件
  cudaEventCreate(&start);  //起始时间
  cudaEventCreate(&stop);   //结束时间

  cudaEventRecord(start, 0);  //记录起始时间
  cspincuda<<<grid, block>>>(gpu_srcData, array_tmp, gpu_sortarray, struct_tmp,self_data,min_data);
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
      // printf("%ld\n",sortarray[i].key);
      // printf("%ld\n",sortarray[i+1].key);
    }
    // printf("%ld\n",sortarray[i].key);
  }
  
  for(int i=0;i<NUM_ELEMENT;i++)
  {
      printf("%ld\n",sortarray[i].key);
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
