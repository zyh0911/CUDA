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
#define GRID_SIZE 2
#define NUM_LISTS 128
unsigned long const NUM_ELEMENT = (1 << 8);

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

__device__ void reduce(unsigned long *const g_idata, unsigned long *const g_odata) {
  //申请共享内存，存在于每个block中 
	__shared__ float partialSum[NUM_LISTS/GRID_SIZE];
 
	//确定索引
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
 
	//传global memory数据到shared memory
	partialSum[tid]=g_idata[i];
 
	//传输同步
	__syncthreads();
	
	//在共享存储器中进行规约
	for(int stride = blockDim.x/2; stride > 0; stride/=2)
	{
		if(tid<stride) partialSum[tid]=min(partialSum[tid+stride],partialSum[tid]);
		__syncthreads();
	}
	//将当前block的计算结果写回输出数组
	if(tid==0)  
		g_odata[blockIdx.x] = partialSum[0];
  /*
  __shared__ unsigned long sdata[64];
  unsigned int tid = threadIdx.x+threadIdx.y*blockDim.x;
  unsigned int i = blockIdx.x * blockDim.x*blockDim.y + tid;
  unsigned int gridSize = blockDim.x*blockDim.y * gridDim.x;
  //printf("%d\n",tid);
  sdata[tid] = 0;

  while (i < blockDim.x*blockDim.y/2) {
    sdata[tid] = min(sdata[tid], min(g_idata[i], g_idata[i + blockDim.x*blockDim.y/2]));
    i += gridSize;
  }

  __syncthreads();

  if (blockDim.x*blockDim.y>= 64) {
    if (tid < 32) {
      sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    }
    __syncthreads();
  }
  if (tid < 32) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
  */
}

#define REDUCTION_SIZE 8
#define REDUCTION_SHIFT 3

__device__ void merge(unsigned long *const data, unsigned long *const array_tmp,
                      const unsigned int tid)  //分块的归约合并
{
  unsigned int index[NUM_LISTS];
  unsigned long self_data[NUM_LISTS];
  index[tid] = 0;
  unsigned long min_data[ GRID_SIZE]={0xFFFFFFFF,0xFFFFFFFF};
  __syncthreads();
  for (int i = 0; i < 1; i++) {
    if (tid + index[tid] * NUM_LISTS < NUM_ELEMENT) {
      self_data[tid] = data[tid + index[tid] * NUM_LISTS];
    } else {
      self_data[tid] = 0xFFFFFFFF;
    }
    __syncthreads();
    reduce(self_data, min_data);
    array_tmp[i] = min(min_data[0],min_data[1]);
    printf("%ld\n",array_tmp[0]);
    for(int j=0;j<NUM_LISTS;j++)
    {
    if(self_data[j]==array_tmp[i])
    {  
      index[j] = index[j] + 1;
      //printf("%d\n",index[tid]);
      //printf("123\n");
    }
    __syncthreads();
    }
  }
  
}

/*
  unsigned int index[NUM_LISTS];
  unsigned int self_data[NUM_LISTS];

  unsigned int min_data[GRID_SIZE];
  unsigned int min_tid[GRID_SIZE];

  unsigned int min_tid_new;
  unsigned int min_value;

  self_data[tid] = 0xFFFFFFFF;
  index[tid] = 0;
  __syncthreads();
  for (int j = 0; j < GRID_SIZE; j++) {
    min_data[j]= 0xFFFFFFFF;
    min_tid[j]= 0xFFFFFFFF;
  }
  min_value=0xFFFFFFFF;
  min_tid_new=0xFFFFFFFF;
  for (int i = 0; i < NUM_ELEMENT; i++) {

    unsigned int block_tid=threadIdx.x + threadIdx.y*blockDim.x;

      if (tid + index[tid] * NUM_LISTS < NUM_ELEMENT) {
        self_data[tid] = data[tid + index[tid] * NUM_LISTS];
      }
      else {
        self_data[tid] = 0xFFFFFFFF;
      }
    for(int j=0;j<GRID_SIZE;j++)
    {
      for(int k=0;k<NUM_LISTS/GRID_SIZE;k++)
      {
        min_data[j]=min(min_data[j], self_data[k]);
      }
      if (self_data[block_tid] == min_value) {
        min_tid[j] = min(min_tid[j], block_tid+j*NUM_LISTS/GRID_SIZE);
      }
    }
    for (int j = 0; j < GRID_SIZE; j++) {
    min_value=min(min_value,min_data[j]);
    }
    __syncthreads();
    for (int j = 0; j < GRID_SIZE; j++) {
    if (min_data[j] == min_value) {
      min_tid_new= min(min_tid_new,min_tid[j]);
    }}
    __syncthreads();
    if (tid== min_tid_new) {
      array_tmp[i] = min_value;
      index[tid] = index[tid] + 1;
    }
    __syncthreads();
  }
  */

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
                          sorta *struct_tmp) {
  const unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned int tid = ix + iy * (gridDim.x * blockDim.x);
  copy_index(sortarray, data, tid);  // step1:copy index
  radix_sort(data, array_tmp, tid);
  // merge(data, array_tmp, tid);
  merge(data, array_tmp, tid);

  // sort_index(sortarray, array_tmp, data, tid);         // step2:sort_by_key
  // sort_struct(array_tmp, sortarray, struct_tmp, tid);  // step3:sort array
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

  cudaMalloc((void **)&gpu_srcData, sizeof(unsigned long) * NUM_ELEMENT);
  cudaMalloc((void **)&array_tmp, sizeof(unsigned long) * NUM_ELEMENT);
  cudaMalloc((sorta **)&gpu_sortarray, sizeof(sorta) * NUM_ELEMENT);
  cudaMalloc((sorta **)&struct_tmp, sizeof(sorta) * NUM_ELEMENT);

  cudaMemcpy(gpu_sortarray, sortarray, sizeof(sorta) * NUM_ELEMENT,
             cudaMemcpyHostToDevice);

  // cudaError_t error = cudaGetLastError();
  dim3 grid(GRID_SIZE);
  dim3 block(NUM_LISTS/GRID_SIZE);

  cudaEvent_t start, stop;  //定义事件
  cudaEventCreate(&start);  //起始时间
  cudaEventCreate(&stop);   //结束时间

  cudaEventRecord(start, 0);  //记录起始时间
  cspincuda<<<grid, block>>>(gpu_srcData, array_tmp, gpu_sortarray, struct_tmp);
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
      // printf("%ld\n",sortarray[i].key);
      // printf("%ld\n",sortarray[i+1].key);
    }
    // printf("%ld\n",sortarray[i].key);
  }
  /*for (int i = 0; i < NUM_ELEMENT; i++) {
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
