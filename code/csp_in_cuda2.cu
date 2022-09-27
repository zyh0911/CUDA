// csp in cuda 为了解决多block的问题，计划采用原子操作lock的方法
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
#define GRID_SIZE 2
#define NUM_LISTS 16
unsigned long const NUM_ELEMENT = (1 << 8);

template <class T>
void c_swap(T &x, T &y) {
  T tmp1 = x;
  x = y;
  y = tmp1;
}

struct Lock {
    int *mutex;
    Lock(void) {
        int state = 0;
        cudaMalloc((void **) &mutex, sizeof(int));
        cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }
    ~Lock(void) {
        cudaFree(mutex);
    }
    __device__ void lock(void) {
        while (atomicCAS(mutex, 0, 1) != 0);
    }
    __device__ void unlock(void) {
        atomicExch(mutex, 0);
    }
};

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

__device__ void reduce(unsigned long *const g_idata,
                       unsigned int *const g_odata) {
  __shared__ float partialSum[NUM_LISTS / GRID_SIZE];

  //确定索引
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  //传global memory数据到shared memory
  partialSum[tid] = g_idata[i];

  //传输同步
  __syncthreads();

  //在共享存储器中进行规约
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      partialSum[tid] = min(partialSum[tid + stride], partialSum[tid]);
    __syncthreads();
  }
  //将当前block的计算结果写回输出数组
  if (tid == 0) g_odata[blockIdx.x] = partialSum[0];
  __syncthreads();
}

__global__ void merge(unsigned long *const data, unsigned long *const array_tmp,
                      const unsigned int tid,Lock myLock)  //分块的归约合并
{

  unsigned int index[GRID_SIZE][NUM_LISTS / GRID_SIZE];
  unsigned long self_data[NUM_LISTS];

  __shared__ unsigned int min_value;
  unsigned int min_data[GRID_SIZE];
  self_data[tid] = data[tid];
  index[blockIdx.x][threadIdx.x] = 0;
  __syncthreads();
  for (int i = 0; i < 2; i++) {
    min_value = 4567894;
    __syncthreads();
    // printf("%d is %ld\n",tid,self_data[tid]);
    reduce(self_data, min_data);
    //printf("%d\t",min_data[0]);
/*
    if (i == 1) {
      printf("%d\n", min_data[0]);
      printf("%d\n", min_data[1]);
    }
*/
    for (int j = 0; j < GRID_SIZE; j++) {
      min_value = min(min_value, min_data[j]);
    }
   // if (i == 1) printf("%d\n", min_value);
    //printf("%d\n", min_value);
    array_tmp[i] = min_value;
    __syncthreads();
    myLock.lock();
    if (self_data[tid] == min_value) {
      printf("%d\n",tid);
      printf("%d\n", blockIdx.x);
      printf("%d\n", threadIdx.x);
      index[blockIdx.x][threadIdx.x] = index[blockIdx.x][threadIdx.x] + 1;
      if (index[blockIdx.x][threadIdx.x] < NUM_ELEMENT / NUM_LISTS) {
        self_data[tid] =
            data[threadIdx.x + index[blockIdx.x][threadIdx.x] * NUM_LISTS];
      } else {
        self_data[tid] = 0xFFFFFFFF;
      }
    }
    myLock.unlock();
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
                          sorta *struct_tmp,Lock myLock) {
  const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int tid = ix + iy * (gridDim.x * blockDim.x);
  copy_index(sortarray, data, tid);
  radix_sort(data, array_tmp, tid);

  //merge(data, array_tmp, tid,myLock);
  unsigned int index[GRID_SIZE][NUM_LISTS / GRID_SIZE];
  unsigned long self_data[NUM_LISTS];

  __shared__ unsigned int min_value;
  unsigned int min_data[GRID_SIZE];
  self_data[tid] = data[tid];
  index[blockIdx.x][threadIdx.x] = 0;
  __syncthreads();
  for (int i = 0; i < 2; i++) {
    min_value = 4567894;
    __syncthreads();
    reduce(self_data, min_data);
    for (int j = 0; j < GRID_SIZE; j++) {
      min_value = min(min_value, min_data[j]);
    }
    array_tmp[i] = min_value;
    __syncthreads();
   
    if (self_data[tid] == min_value) {
      index[blockIdx.x][threadIdx.x] = index[blockIdx.x][threadIdx.x] + 1;
      if (index[blockIdx.x][threadIdx.x] < NUM_ELEMENT / NUM_LISTS) {
        self_data[tid] =
            data[threadIdx.x + index[blockIdx.x][threadIdx.x] * NUM_LISTS];
      } else {
        self_data[tid] = 0xFFFFFFFF;
      }
    }

  }
  //sort_index(sortarray, array_tmp, data, tid);

 // sort_struct(array_tmp, sortarray, struct_tmp, tid);
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

  Lock myLock;
  cudaMalloc((void **)&gpu_srcData, sizeof(unsigned long) * NUM_ELEMENT);
  cudaMalloc((void **)&array_tmp, sizeof(unsigned long) * NUM_ELEMENT);
  cudaMalloc((sorta **)&gpu_sortarray, sizeof(sorta) * NUM_ELEMENT);
  cudaMalloc((sorta **)&struct_tmp, sizeof(sorta) * NUM_ELEMENT);

  cudaMemcpy(gpu_sortarray, sortarray, sizeof(sorta) * NUM_ELEMENT,
             cudaMemcpyHostToDevice);

  // cudaError_t error = cudaGetLastError();
  dim3 grid(GRID_SIZE);
  dim3 block(NUM_LISTS / GRID_SIZE);

  cudaEvent_t start, stop;  //定义事件
  cudaEventCreate(&start);  //起始时间
  cudaEventCreate(&stop);   //结束时间

  cudaEventRecord(start, 0);  //记录起始时间

  cspincuda<<<grid, block>>>(gpu_srcData, array_tmp, gpu_sortarray, struct_tmp, myLock);
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

  int result = 0;
  for (int i = 0; i < NUM_ELEMENT - 1; i++) {
    if (sortarray[i].key != sortarray[i + 1].key - 1) {
      result++;
    }
  }
/*  
  for (int i = 0; i < NUM_ELEMENT; i++) {
    printf("%ld\n", sortarray[i].key);
  }
*/
  printf("CUDA error: %s\n", cudaGetErrorString(error));
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
