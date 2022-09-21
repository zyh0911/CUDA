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
#define NUM_ELEMENT 1048576
int num_data = NUM_ELEMENT;

template<class T> void c_swap(T &x, T &y){ T tmp1 = x; x = y; y = tmp1; }


int main(void)
{
    struct sorta{
        unsigned long key;
    };
    
    unsigned long ig=0;
    sorta sortarray[NUM_ELEMENT];
    
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
    struct seed_compare
    {
        __host__ __device__
        bool operator()(const sorta& a, const sorta& b)
        {
                if(a.key< b.key)
                {
                    return true;
                }
                else 
                {
                    return false;
                }
        }
    };
    start = clock();
    thrust::sort(thrust::device,sortarray,sortarray+NUM_ELEMENT,seed_compare());
    end = clock();   
   for(int i=0;i<num_data;i++)
    {
        printf("%ld\n",sortarray[i].key);
       
    }
    printf("run time is %.8lf\n", (double)(end-start)/1000000);
    
}
