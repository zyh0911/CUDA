//merge sort
#include<iostream>
#include<math.h>
#include<cuda_runtime.h>
#include"device_launch_parameters.h" 
#include<fstream>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
using namespace std;
unsigned long const  NUM_ELEMENT=(1<<7);
template<class T> void c_swap(T &x, T &y){ T tmp1 = x; x = y; y = tmp1; }

typedef struct SORTSTRUCT{
    unsigned long key;
    } sorta;

bool seed_compare(SORTSTRUCT a, SORTSTRUCT b)
    {
        return a.key< b.key;
    };
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
    cudaMallocManaged((void **)&sortarray,sizeof(sortarray)*NUM_ELEMENT);
    start = clock();
    thrust::sort(sortarray,sortarray+NUM_ELEMENT,seed_compare);
    end = clock();

    cudaError_t error = cudaGetLastError();   
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    
    
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
