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
unsigned long const  NUM_ELEMENT=(1<<20);
template<class T> 
void c_swap(T &x, T &y)
{   T tmp1 = x;
    x = y; 
    y = tmp1; 
}

template <typename S> 
void sort_long_arrays(S *array, int n) { 
 int *keys; // the buffer for keys
 int *indices; // the buffer for indices
 S *tmp; // the buffer for permutation
 // step 0: properly allocate the buffers
 cudaMallocManaged((void **)&keys,sizeof(keys)*NUM_ELEMENT);
 cudaMallocManaged((void **)&indices,sizeof(indices)*NUM_ELEMENT);
 cudaMallocManaged((void **)&tmp,sizeof(tmp)*NUM_ELEMENT);
 // step 1: Copy
 for (int i = 0; i < n; ++i) {
 keys[i] = array[i].key;
 indices[i] = i;
 }
 // step 2: Sort
   thrust::sort_by_key(keys,keys+n, indices);
   thrust::copy_n(thrust::make_permutation_iterator(array, indices),n, tmp);
   thrust::copy_n(tmp, n, array);
 }

typedef struct SORTSTRUCT{
    unsigned long key;
    }sorta;
sorta sortarray[NUM_ELEMENT];

int main(void)
{
    for(unsigned long i = 0; i < NUM_ELEMENT; i++)  
    {
        sortarray[i].key = i;
    }  
    for(int i = 0; i < NUM_ELEMENT; i++)
    {
        c_swap(sortarray[rand()%7].key, sortarray[i].key);
    }

    clock_t start, end;
    start = clock();
    sort_long_arrays(sortarray, NUM_ELEMENT);
    end = clock();   

    int result=0;
    for(int i=0;i<NUM_ELEMENT-1;i++)
    {
        if(sortarray[i].key>sortarray[i+1].key)
        {
            result++;
            //printf("%ld\n",sortarray[i].key);
            //printf("%ld\n",sortarray[i+1].key);
        }
        //printf("%ld\n",sortarray[i].key);
    }
    /*
    for(int i=0;i<NUM_ELEMENT;i++)
    {
        printf("%ld\n",sortarray[i].key);
    }
    */
    printf("%ld\n",NUM_ELEMENT);
    printf("%d\n",result);
    if(result==0)
    {
        printf("result is true.\n");
    }
    else
    {
        printf("result is false.\n");
    }
    printf("run time is %.8lf\n", (double)(end-start)/CLOCKS_PER_SEC);
    
}



