//csp in cuda new
#include<iostream>
#include<math.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<fstream>
#include<algorithm>
#include<thrust/sort.h>
#include<thrust/execution_policy.h>
#include<stdio.h>
#include<stdlib.h>

using namespace std;

#define NUM_ELEMENT 524288
#define NUM_LISTS   1024
#define REDUCTION_SIZE  8
#define REDUCTION_SHIFT 3

template<class T> 
void c_swap(T &x, T &y)
{   T tmp1 = x; 
    x = y;
    y = tmp1;
}

template <typename S> 
__device__ void copy_index(S * sortarray,\
                unsigned long * const data,\
                const unsigned int tid)
{
    for(int i = 0; i < NUM_ELEMENT;  i+=NUM_LISTS)
    {
        data[tid+i]=sortarray[tid+i].key; 
    }
    __syncthreads();
}

__device__ void radix_sort(unsigned long * const sort_tmp,\
                            unsigned long  * const sort_tmp_1,\
                            const unsigned int tid) 
{
    
    for(unsigned long bit_mask = 1; bit_mask > 0; bit_mask <<= 1)    
    {
        unsigned int base_cnt_0 = 0;
        unsigned int base_cnt_1 = 0;

        for (int i = 0; i < NUM_ELEMENT; i+=NUM_LISTS) 
        {
            if(sort_tmp[i+tid] & bit_mask)  
            {
                sort_tmp_1[base_cnt_1+tid] = sort_tmp[i+tid];
                base_cnt_1 += NUM_LISTS;
            }
            else    
            {
                sort_tmp[base_cnt_0+tid] = sort_tmp[i+tid];
                base_cnt_0 += NUM_LISTS;
            }
        }

        for (unsigned long i = 0; i < base_cnt_1; i+=NUM_LISTS)  
        {
            sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
        }
        __syncthreads();
    }
}

__device__ void merge(      unsigned long * const data,\
                            unsigned long  * const array_tmp,\
                            const unsigned int tid)
{  
    __shared__ unsigned  int index[NUM_LISTS];
    __shared__ unsigned int min_data;
    
    index[tid]=0;
    __syncthreads();
    
    for(long i = 0; i < NUM_ELEMENT; i++)
    {
        __shared__ unsigned long self_data[NUM_LISTS];
        self_data[tid]=0xFFFFFFFF;
        min_data=0xFFFFFFFF;
        __syncthreads();
        
        if (tid+index[tid]*NUM_LISTS<NUM_ELEMENT)
        {
            self_data[tid]=data[tid+index[tid]*NUM_LISTS];
        }
        else
        {
            self_data[tid] = 0xFFFFFFFF;
        }
        __syncthreads();
        atomicMin(&(min_data), self_data[tid]);  
        __syncthreads();
        if(self_data[tid]==min_data)
        {
            array_tmp[i]=min_data;
            index[tid]=index[tid]+1;
        }
        __syncthreads();
        
    }
    
}
template <typename S> 
__device__ void sort_index( S* sortarray,\
                            unsigned long * const array_tmp,\
                            unsigned long * const data,\
                            const unsigned int tid)
{
     for(int i = 0; i < NUM_ELEMENT;  i+=NUM_LISTS)
    {
        data[tid+i]=search(sortarray,array_tmp[i+tid]);
    }
    __syncthreads();
}

template <typename S> 
__device__ void  sort_struct(   unsigned long * const data,\
                                S* sortarray,\
                                S* struct_tmp,\
                                const unsigned int tid)
{
    for(int i = 0; i < NUM_ELEMENT;  i+=NUM_LISTS)
    {
        struct_tmp[tid+i]=sortarray[data[tid+i]];
    }
    __syncthreads();
    for(int i = 0; i < NUM_ELEMENT;  i+=NUM_LISTS)
    {
        sortarray[tid+i]=struct_tmp[tid+i];
    }
    __syncthreads();
}

template <typename S> 
__device__ int search(S *nums, unsigned long val)
{
    
    for(int i=0;i<NUM_ELEMENT;i++)
    {
        if(nums[i].key==val)
        {    
            return i;
        }
    }
    return -1;
}
typedef struct SORTSTRUCT{
    unsigned long key;
    }sorta;

__global__ void cspincuda(  unsigned long * const data,\
                            unsigned long * const array_tmp,\
                            sorta * sortarray,\
                            sorta * struct_tmp)
{   
    const unsigned int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
    const unsigned int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    const unsigned int tid = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;
    
    copy_index(sortarray,data,tid);//step1:copy index
    radix_sort(data,array_tmp,tid);
    merge( data, array_tmp,tid);
    sort_index(sortarray,array_tmp,data,tid);//step2:sort_by_key
    sort_struct(data,sortarray,struct_tmp,tid);//step3:sort array

}


int main(void)
{   
    
    sorta sortarray[NUM_ELEMENT];
    for(unsigned long i = 0; i < NUM_ELEMENT; i++)  
    {
        sortarray[i].key = i;
    }  
    for(int i = 0; i < NUM_ELEMENT; i++)
    {
        c_swap(sortarray[rand()%NUM_ELEMENT].key, sortarray[i].key);
    }
    
    unsigned long  *gpu_srcData;
    unsigned long  *array_tmp;
    sorta * gpu_sortarray;
    sorta * struct_tmp;
    
    cudaMalloc((void**)&gpu_srcData, sizeof(unsigned long)*NUM_ELEMENT);
    cudaMalloc((void**)&array_tmp, sizeof(unsigned long)*NUM_ELEMENT);
    cudaMalloc((sorta**)&gpu_sortarray, sizeof(sorta)*NUM_ELEMENT);
    cudaMalloc((sorta**)&struct_tmp, sizeof(sorta)*NUM_ELEMENT);

    cudaMemcpy(gpu_sortarray, sortarray, sizeof(sorta)*NUM_ELEMENT, cudaMemcpyHostToDevice);
    
    //cudaError_t error = cudaGetLastError();
    dim3 grid(1);
    dim3 block(512,2);  

    clock_t start, end;
    start = clock();
    cspincuda<<<grid,block>>>(gpu_srcData,array_tmp,gpu_sortarray,struct_tmp);
    
      
    
    cudaMemcpy(sortarray, gpu_sortarray, sizeof(sorta)*NUM_ELEMENT, cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    cudaDeviceSynchronize();
    end = clock();
    /*
    for(int i=0;i<NUM_ELEMENT;i++)
    {
        printf("%ld\n",sortarray[i].key);
        
    }
    */
    printf("run time is %.8lf\n", (double)(end-start)/CLOCKS_PER_SEC);
    
}



