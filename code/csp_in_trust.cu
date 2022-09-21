//CSP IN THRUST
#include<iostream>
#include<math.h>
#include<cuda_runtime.h>
#include"device_launch_parameters.h" 
#include<fstream>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;
#define NUM_ELEMENT 1048576
int num_data = NUM_ELEMENT;


template<class T> void c_swap(T &x, T &y){ T tmp1 = x; x = y; y = tmp1; }

template <typename S> 
void sort_long_arrays(S *array, int n) { 
 int *keys; // the buffer for keys
 int *indices; // the buffer for indices
 S *tmp; // the buffer for permutation
 // step 0: properly allocate the buffers
 cudaMallocManaged((void **)&keys,sizeof(keys)*num_data);
 cudaMallocManaged((void **)&indices,sizeof(indices)*num_data);
 cudaMallocManaged((void **)&tmp,sizeof(tmp)*num_data);
 // step 1: Copy
 for (int i = 0; i < n; ++i) {
 keys[i] = array[i].key;
 indices[i] = i;
 }
 // step 2: Sort
 thrust::sort_by_key(thrust::device,keys, keys+n, indices);
 thrust::copy_n(thrust::device,thrust::make_permutation_iterator(array, indices),n, tmp);
 thrust::copy_n(thrust::device,tmp, n, array);
 }

int main(void)
{
    struct sorta{
        unsigned long key;
    };
    unsigned long ig=0;
    struct sorta sortarray[NUM_ELEMENT];
    for(int i = 0; i < num_data; i++)  
    {
        sortarray[i].key = ig;
        ig=ig+1;
    }  

    for(int i = 0; i < num_data; i++)
    {
        c_swap(sortarray[rand()%num_data].key, sortarray[i].key);
    }

    clock_t start, end;
    start = clock();
    sort_long_arrays(sortarray, num_data);
    end = clock();   
   for(int i=0;i<num_data;i++)
    {
        printf("%ld\n",sortarray[i].key);
       
    }
    printf("run time is %.8lf\n", (double)(end-start)/1000000);
    
}



