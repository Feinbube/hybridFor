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

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11. 
 */

 // Sources:
 // [1] http://developer.nvidia.com/cuda-cc-sdk-code-samples --> Matrix Multiplication (CUDA Driver API version with Dynamic Linking Version)

 //Usage:
 // Command line parameters:
	// aw : width of matrix A
	// ah : height of matrix A
	// bw : width of matrix B
	// w : number of warmup rounds
	// m : number of measured rounds
	// c : number of cooldown rounds
	
	// example: -m=10 => 10 rounds with measured time are executed
	
	// -cublas : if this parameter is set only the cublas implementation is run, otherwise CUDA and CUBLAS implementations are run
	  

// Utilities and system includes
#include <cublas_v2.h>
#include <shrUtils.h>
#include <shrQATest.h>
#include "cutil_inline.h"
#include "matrixMul.h"

// includes, kernels
#include <matrixMul_kernel.cu>

static char *sSDKsample = "matrixMul";

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

int numWarmupRounds = 5;
int numMeasuredRounds = 3;
int numCoolDownRounds = 2;

void inline checkError(cublasStatus_t status, const char* msg)
{
    if(status != CUBLAS_STATUS_SUCCESS){
        printf(msg);
        exit(-1);
    }
}

int aw = 0;
int ah = 0;
int bw = 0;
bool sizeSet = false;

void processCommandLine(int argc, char **argv)
{
	int cmdVali = 0;
	if (cutGetCmdLineArgumenti( argc, (const char**)argv, "aw", &cmdVali))
	{
		aw = cmdVali;
		printf("The width of the matrix A is set to: %d.\nNote: The width is set default for all dimensions of the matrices A x B = C.\n", aw);
		ah = aw;
		bw = aw;
		sizeSet = true;
	}

	if (cutGetCmdLineArgumenti( argc, (const char**)argv, "ah", &cmdVali))
	{
		ah = cmdVali;
		printf("The height of matrix A is set to: %d.\nNote: The parameter aw for the width of matrix A must also be set.\n", ah);
	}

	if (cutGetCmdLineArgumenti( argc, (const char**)argv, "bw", &cmdVali))
	{
		bw = cmdVali;
		printf("The width of the matrix B is set to: %d.\nNote: The parameter aw for the width of matrix A must also be set.\n", bw);
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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    shrQAStart(argc, argv);
	printf("[ %s ]\n", sSDKsample);

    //shrSetLogFileName ("matrixMul.txt");
    shrLog("%s Starting (CUDA and CUBLAS tests)...\n\n", argv[0]);
	processCommandLine(argc, argv);
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
    if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    {
        cutilDeviceInit(argc, argv);
    }
    else
    {
        cutilSafeCall( cudaSetDevice(cutGetMaxGflopsDeviceId()) );
    }

    int devID;
    cudaDeviceProp props;

    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDevice(&devID));
    cutilSafeCall(cudaGetDeviceProperties(&props, devID));

    // use a larger block size for Fermi and above
    int block_size = (props.major < 2) ? 16 : 32;

    printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	// set seed for rand()
    srand(2006);

    // Optional Command-line multiplier for matrix sizes
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
    int iSizeMultiple = 5;
    shrGetCmdLineArgumenti(argc, (const char**)argv, "sizemult", &iSizeMultiple); 
    iSizeMultiple = CLAMP(iSizeMultiple, 1, 10);

    bool useCublasOnly = false;
    if(shrCheckCmdLineFlag(argc, (const char**)argv, "cublas"))
        useCublasOnly = true;

	// For GPUs with fewer # of SM's, we limit the maximum size of the matrix
	if (props.multiProcessorCount <= 0 /* 4 is the original value */) {
		uiWA = 2 * block_size * iSizeMultiple;
		uiHA = 4 * block_size * iSizeMultiple;
		uiWB = 2 * block_size * iSizeMultiple;
		uiHB = 4 * block_size * iSizeMultiple;
		uiWC = 2 * block_size * iSizeMultiple;
		uiHC = 4 * block_size * iSizeMultiple;
	} else {
		if (sizeSet)
		{
			uiWA = aw;
			uiHA = ah;
			uiWB = bw;
			uiHB = aw;
			uiWC = bw;
			uiHC = ah;
		} else {
			uiWA = WA * iSizeMultiple;
			uiHA = HA * iSizeMultiple;
			uiWB = WB * iSizeMultiple;
			uiHB = HB * iSizeMultiple;
			uiWC = WC * iSizeMultiple;
			uiHC = HC * iSizeMultiple;
		}
	}
    shrLog("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n\n", 
            uiWA, uiHA, uiWB, uiHB, uiWC, uiHC);

    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    
    // allocate device memory
    float* d_A, *d_B, *d_C;
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float* h_C      = (float*) malloc(mem_size_C);
	float* h_CUBLAS = (float*) malloc(mem_size_C);

    cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
    cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_B));

    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
    cutilSafeCall(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );
    
    cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_C));
   
    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(uiWC / threads.x, uiHC / threads.y);

    // kernel warmup
        
    // create and start timer
    shrLog("Runing Kernels...\n\n");
    unsigned int timer_cublas    = 0;
    unsigned int timer_matrixMul = 0;

    // execute the kernel
   	
	// CUBLAS version 2.0
	{
        cublasHandle_t handle;
        checkError(cublasCreate(&handle), "cublasCreate() error!\n");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        //Perform warmup operation with cublas
		for (int i = 0; i < numWarmupRounds; i++)
		{
        cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiWB, uiHA, uiWA, &alpha, d_B, uiWB, d_A, uiWA, &beta, d_C, uiWB);
        checkError(ret, "cublas Sgemm returned an error!\n");
		}

		//Perform measured rounds

		float overallTime = 0.0f;
		cutilCheckError(cutCreateTimer(&timer_cublas));
		for (int i = 0; i < numMeasuredRounds; i++)
		{
			cutilSafeCall( cutilDeviceSynchronize() );
			cutilCheckError( cutResetTimer(timer_cublas) );
			cutilCheckError( cutStartTimer(timer_cublas) );

			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiWB, uiHA, uiWA, &alpha, d_B, uiWB, d_A, uiWA, &beta, d_C, uiWB);

			// check if kernel execution generated and error
			cutilCheckMsg("CUBLAS Kernel execution failed");
			cutilDeviceSynchronize();
			// stop and destroy timer
			cutilCheckError(cutStopTimer(timer_cublas));
			float time = cutGetTimerValue(timer_cublas);
			overallTime += time;
			shrLog("The elapsed time for round %d is: %f ms.\n", i+1, time);
		}

		for (int i = 0; i < numCoolDownRounds; i++)
		{
        cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, uiWB, uiHA, uiWA, &alpha, d_B, uiWB, d_A, uiWA, &beta, d_C, uiWB);
        checkError(ret, "cublas Sgemm returned an error!\n");
		}

		overallTime /= numMeasuredRounds;	
		printf("The average elapsed time is: %f ms.\n\n", overallTime);

		cutilCheckError(cutDeleteTimer(timer_cublas));

		// copy result from device to host
		cutilSafeCall(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

        checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
	}

	// For the case where "-cublas" is not specified, we will run the matrixMul kernel
	if (!useCublasOnly) 
	{
		//Performs warmup operation using matrixMul CUDA kernel
		for (int i = 0; i < numWarmupRounds; i++)
		{
			if (block_size == 16) {
				matrixMul<16><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
			} else {
				matrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
			}
		}

		float overallTime = 0.0f;
		cutilCheckError(cutCreateTimer(&timer_matrixMul));
		for (int i = 0; i < numMeasuredRounds; i++)
		{
			cutilSafeCall( cutilDeviceSynchronize() );
			cutilCheckError( cutResetTimer(timer_matrixMul) );
			cutilCheckError( cutStartTimer(timer_matrixMul) );
			if (block_size == 16) {
				matrixMul<16><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
			} else {
				matrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
			}
			// check if kernel execution generated and error
			cutilCheckMsg("CUDA matrixMul Kernel execution failed");
			cutilDeviceSynchronize();
			// stop and destroy timer
			cutilCheckError(cutStopTimer(timer_matrixMul));

			float time = cutGetTimerValue(timer_matrixMul);
			overallTime += time;
			shrLog("The elapsed time for round %d is: %f ms.\n", i+1, time);
		}

		for (int i = 0; i < numCoolDownRounds; i++)
		{
			if (block_size == 16) {
				matrixMul<16><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
			} else {
				matrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
			}
		}

		overallTime /= numMeasuredRounds;
		shrLog("The average elapsed time is: %f ms.\n\n", overallTime);
		shrLogEx(LOGBOTH | MASTER, 0, "NumDevsUsed = %d, Workgroup = %u\n", 1, threads.x * threads.y);

		cutilCheckError(cutDeleteTimer(timer_matrixMul));

		// copy result from device to host
		cutilSafeCall(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );
	}

    // compute reference solution
    shrLog("\nComparing GPU results with Host computation...\n\n");    
    float* reference = (float*)malloc(mem_size_C);
    computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

    // check result (CUBLAS)
	printf("Comparing CUBLAS & Host results\n");
    shrBOOL resCUBLAS = shrCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f);
    if (resCUBLAS != shrTRUE) 
    {
        printDiff(reference, h_CUBLAS, uiWC, uiHC, 100, 1.0e-5f);
    }
    shrLog("CUBLAS compares %s\n\n", (shrTRUE == resCUBLAS) ? "OK" : "FAIL");

    // check result (matrixMul)
	shrBOOL resCUDA = shrTRUE;
	if (!useCublasOnly)
	{
		printf("Comparing CUDA matrixMul & Host results\n");
		resCUDA = shrCompareL2fe(reference, h_C, size_C, 1.0e-6f);
		if (resCUDA != shrTRUE) 
		{
			printDiff(reference, h_C, uiWC, uiHC, 100, 1.0e-5f);
		}
		shrLog("CUDA matrixMul compares %s\n\n", (shrTRUE == resCUDA) ? "OK" : "FAIL");
	}
    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cutilSafeCall(cudaFree(d_A));
    cutilSafeCall(cudaFree(d_B));
    cutilSafeCall(cudaFree(d_C));

    cutilDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (resCUDA == shrTRUE && resCUBLAS == shrTRUE) ? QA_PASSED : QA_FAILED);
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    shrLog("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength)
        {
            shrLog("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) 
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) 
            {                
                if (error_count < iListLength)
                {
                    shrLog("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    shrLog(" \n  Total Errors = %d\n\n", error_count);
}
