/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application, doesn't use cutil library.
*/

#include <stdio.h>
#include <string.h>
#include <iostream>

#include <shrQATest.h>

using namespace std;

bool g_bQATest = false;

#ifdef _WIN32
   #define STRCASECMP  _stricmp
   #define STRNCASECMP _strnicmp
#else
   #define STRCASECMP  strcasecmp
   #define STRNCASECMP strncasecmp
#endif

#define ASSERT(x, msg, retcode) \
    if (!(x)) \
    { \
        cout << msg << " " << __FILE__ << ":" << __LINE__ << endl; \
        return retcode; \
    }

inline cudaError cutilDeviceSynchronize()
{
#if CUDART_VERSION >= 4000
	return cudaDeviceSynchronize();
#else
	return cudaThreadSynchronize();
#endif
}

inline void cutilDeviceReset()
{
#if CUDART_VERSION >= 4000
	cutilDeviceReset();
#else
	cudaThreadExit();
#endif
}

__global__ void sequence_gpu(int *d_ptr, int length)
{
    int* ptr = (int*)malloc(10*sizeof(int));
	int sum = 0;
	for (int i = 0; i < 10; i++)
	{
		ptr[i]= i; 
	}

	for (int j = 0; j < 10; j++)
	{
		sum = sum + ptr[j];
	}
	free(ptr);
	int elemID = blockIdx.x * blockDim.x + threadIdx.x;
    if (elemID < length)
    {
        d_ptr[elemID] = elemID + sum;
    }
	
}


void sequence_cpu(int *h_ptr, int length)
{
    for (int elemID=0; elemID<length; elemID++)
    {
        h_ptr[elemID] = elemID;
    }
}

void processArgs(int argc, char **argv)
{
    for (int i=1; i < argc; i++) {
        if((!STRNCASECMP((argv[i]+1), "noprompt", 8)) || (!STRNCASECMP((argv[i]+2), "noprompt", 8)) )
        {
            g_bQATest = true;
        }
    }
}

int main(int argc, char **argv)
{
	shrQAStart(argc, argv);

    cout << "CUDA Runtime API template" << endl;
    cout << "=========================" << endl;
    cout << "Self-test started" << endl;

    const int N = 1024;

    processArgs(argc, argv);

    int *d_ptr;
    ASSERT(cudaSuccess == cudaMalloc(&d_ptr, N * sizeof(int)), "Device allocation of " << N << " ints failed", -1);

    int *h_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_ptr, N * sizeof(int)), "Host allocation of " << N << " ints failed", -1);

    cout << "Memory allocated successfully" << endl;

    dim3 cudaBlockSize(32,1,1);
    dim3 cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
	cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

    sequence_gpu<<<cudaGridSize, cudaBlockSize>>>(d_ptr, N);
    ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
    ASSERT(cudaSuccess == cutilDeviceSynchronize(), "Kernel synchronization failed", -1);

    sequence_cpu(h_ptr, N);

    cout << "CUDA and CPU algorithm implementations finished" << endl;

    int *h_d_ptr;
    ASSERT(cudaSuccess == cudaMallocHost(&h_d_ptr, N * sizeof(int)), "Host allocation of " << N << " ints failed", -1);
    ASSERT(cudaSuccess == cudaMemcpy(h_d_ptr, d_ptr, N * sizeof(int), cudaMemcpyDeviceToHost), "Copy of " << N << " ints from device to host failed", -1);
    bool bValid = true;
    for (int i=0; i<N && bValid; i++)
    {
        if (h_ptr[i] != h_d_ptr[i] - 45)
        {
            bValid = false;
        }
    }

    ASSERT(cudaSuccess == cudaFree(d_ptr), "Device deallocation failed", -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_ptr), "Host deallocation failed", -1);
    ASSERT(cudaSuccess == cudaFreeHost(h_d_ptr), "Host deallocation failed", -1);

    cout << "Memory deallocated successfully" << endl;

    cout << "TEST Results " << endl;
    
    shrQAFinishExit(argc, (const char **)argv, (bValid ? QA_PASSED : QA_FAILED));
}
