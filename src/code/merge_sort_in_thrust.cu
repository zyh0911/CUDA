#include <math.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <src/code/csp.hpp>

namespace src::code {
unsigned long const NUM_ELEMENT = (1 << 20);

void  merge_sort_in_thrust(data*d) {
  cudaMallocManaged((data**)&d, sizeof(data) * NUM_ELEMENT);
  thrust::sort(d, d+ NUM_ELEMENT, seed_compare);
  cudaDeviceSynchronize();
}
// int result=0;
// for(int i=0;i<NUM_ELEMENT-1;i++)
// {
//     if(data[i].key>data[i+1].key)
//     {
//         result++;
//         //printf("%ld\n",sortarray[i].key);
//         //printf("%ld\n",sortarray[i+1].key);
//     }
//     //printf("%ld\n",sortarray[i].key);
// }
/*
for(int i=0;i<NUM_ELEMENT;i++)
{
    printf("%ld\n",sortarray[i].key);
}
*/
// printf("%ld\n",NUM_ELEMENT);
// printf("%d\n",result);
// if(result==0)
// {
//     printf("result is true.\n");
// }
// else
// {
//     printf("result is false.\n");
// }

}  // namespace src::code