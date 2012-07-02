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

/*
 * This sample calculates scalar products of a 
 * given set of input vector pairs
 */

 // Sources
 // [1] http://developer.nvidia.com/cuda-cc-sdk-code-samples --> samples: Scalar Product and Matrix Multiplication (CUDA Runtime API Version)
 // [2] http://developer.download.nvidia.com/compute/DevZone/docs/html/CUDALibraries/doc/CUBLAS_Library.pdf

// Usage:
// Command line parameters:
	// n : number of input elements per Vector
	// w : number of warmup rounds
	// m : number of measured rounds
	// c : number of cooldown rounds
	
	// example: -m=10 => 10 rounds with measured time are executed

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cublas_v2.h>
#include <cutil_inline.h>
#include <shrQATest.h>


///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void scalarProdCPU(
    float *h_C,
    float *h_A,
    float *h_B,
    int elementN
);


////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

//Number of elements per vector; arbitrary, 
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
int ELEMENT_N = 4096;
//Total number of data elements
int   DATA_SZ = ELEMENT_N * sizeof(float);
int RESULT_SZ = sizeof(float);

int numWarmupRounds = 5;
int numMeasuredRounds = 3;
int numCoolDownRounds = 2;


void processCommandLine(int argc, char **argv)
{
	int cmdVali = 0;
	if (cutGetCmdLineArgumenti( argc, (const char**)argv, "n", &cmdVali))
	{
		ELEMENT_N = cmdVali;
		printf("The number of elements is set to: %d\n", ELEMENT_N);
		DATA_SZ = ELEMENT_N * sizeof(float);
		RESULT_SZ = sizeof(float);
	}
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "w", &cmdVali))
	{
		numWarmupRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numWarmupRounds);
	}
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "m", &cmdVali))
	{
		numMeasuredRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numMeasuredRounds);
	} 
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "c", &cmdVali))
	{
		numCoolDownRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numCoolDownRounds);
	}
}

void inline checkError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS){
        printf(msg);
        exit(-1);
    }
}


///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    float *h_A, *h_B, *h_C_CPU, *h_C_GPU;
    float *d_A, *d_B;
    double delta, ref, L1norm;
    unsigned int hTimer;
    int i;
	cublasHandle_t handle;
    checkError(cublasCreate(&handle), "cublasCreate() error!\n");

    shrQAStart(argc, argv);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );

	processCommandLine(argc, argv);

    cutilCheckError( cutCreateTimer(&hTimer) );

    printf("Initializing data...\n");
        printf("...allocating CPU memory.\n");
        h_A     = (float *)malloc(DATA_SZ);
        h_B     = (float *)malloc(DATA_SZ);
        h_C_CPU = (float *)malloc(RESULT_SZ);
        h_C_GPU = (float *)malloc(RESULT_SZ);

        printf("...allocating GPU memory.\n");
        cutilSafeCall( cudaMalloc((void **)&d_A, DATA_SZ)   );
        cutilSafeCall( cudaMalloc((void **)&d_B, DATA_SZ)   );

        printf("...generating input data in CPU mem.\n");
        srand(123);
        //Generating input data on CPU
        for(i = 0; i < ELEMENT_N; i++){
            h_A[i] = RandFloat(0.0f, 1.0f);
            h_B[i] = RandFloat(0.0f, 1.0f);
        }

        printf("...copying input data to GPU mem.\n");
        //Copy options data to GPU memory for further processing 
        cutilSafeCall( cudaMemcpy(d_A, h_A, DATA_SZ, cudaMemcpyHostToDevice) );
        cutilSafeCall( cudaMemcpy(d_B, h_B, DATA_SZ, cudaMemcpyHostToDevice) );
		printf("Data init done.\n");


		printf("Executing GPU kernel...\n");

		for (int i = 0; i < numWarmupRounds; i++)
		{
			cublasSdot(handle, ELEMENT_N, d_A, 1, d_B, 1,  h_C_GPU);
			cutilCheckMsg("scalarProdGPU() execution failed\n");
		}

		float overallTime = 0.0f;
		for (int i = 0; i < numMeasuredRounds; i++)
		{
			cutilSafeCall( cutilDeviceSynchronize() );
			cutilCheckError( cutResetTimer(hTimer) );
			cutilCheckError( cutStartTimer(hTimer) );
			cublasSdot(handle, ELEMENT_N, d_A, 1, d_B, 1,  h_C_GPU);
			cutilCheckMsg("scalarProdGPU() execution failed\n");
			cutilSafeCall( cutilDeviceSynchronize() );
			cutilCheckError( cutStopTimer(hTimer) );
			float time = cutGetTimerValue(hTimer);
			overallTime += time;
			printf("The elapsed time for round %d is: %f ms.\n", i+1, time);
		}

		for(int i = 0; i < numCoolDownRounds; i++)
		{
			cublasSdot(handle, ELEMENT_N, d_A, 1, d_B, 1,  h_C_GPU);
			cutilCheckMsg("scalarProdGPU() execution failed\n");
		}

		overallTime /= numMeasuredRounds;	
	printf("The average elapsed time is: %f ms.\n\n", overallTime);
	
	printf("The result is %f.\n", *h_C_GPU);

    printf("Checking GPU results...\n");
        printf("..running CPU scalar product calculation\n");
        scalarProdCPU(h_C_CPU, h_A, h_B, ELEMENT_N);

        printf("...comparing the results\n");
        //Calculate max absolute difference and L1 distance
        //between CPU and GPU results
        delta = fabs(h_C_GPU[0] - h_C_CPU[0]);
        ref   = h_C_CPU[0];
        L1norm = delta / ref;

    printf("Shutting down...\n");
        cutilSafeCall( cudaFree(d_B)   );
        cutilSafeCall( cudaFree(d_A)   );
        free(h_C_GPU);
        free(h_C_CPU);
        free(h_B);
        free(h_A);
        cutilCheckError( cutDeleteTimer(hTimer) );

    cutilDeviceReset();
    printf("L1 error: %E\n", L1norm);
    shrQAFinishExit(argc, (const char **)argv, (L1norm < 1e-6) ? QA_PASSED : QA_FAILED);
}
