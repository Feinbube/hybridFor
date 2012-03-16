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
 * This application demonstrates how to use multiple GPUs with OpenCL.
 * Event Objects are used to synchronize the CPU with multiple GPUs. 
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the 
 * application. On the other side, you can still extend your desktop to screens 
 * attached to both GPUs.
 */
/*
#include <iostream>
#include <CL/opencl.h>
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const unsigned int MAX_GPU_COUNT = 8;
const unsigned int DATA_N = 1048576*24;
const unsigned int BLOCK_N = 128;
const unsigned int THREAD_N = 128;
const unsigned int ACCUM_N = BLOCK_N * THREAD_N;

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
   
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    // OpenCL
    cl_platform_id cpPlatform;
    cl_uint ciDeviceCount;
    cl_device_id* cdDevices;
    cl_context cxGPUContext;
    cl_device_id cdDevice;                          // GPU device
    int deviceNr[MAX_GPU_COUNT];
    cl_command_queue commandQueue[MAX_GPU_COUNT];
    cl_mem d_Data[MAX_GPU_COUNT];
    cl_mem d_Result[MAX_GPU_COUNT];
    cl_program cpProgram; 
    cl_kernel reduceKernel[MAX_GPU_COUNT];
    cl_event GPUDone[MAX_GPU_COUNT];
    cl_event GPUExecution[MAX_GPU_COUNT];
    size_t programLength;
    cl_int ciErrNum;			               
    char cDeviceName [256];
    cl_mem h_DataBuffer;

    // Vars for reduction results
    float h_SumGPU[MAX_GPU_COUNT * ACCUM_N];   
    float *h_Data;
    double sumGPU;
    double sumCPU, dRelError;

    // allocate and init host buffer with with some random generated input data
    h_Data = (float *)malloc(DATA_N * sizeof(float));
    // shrFillArray(h_Data, DATA_N);
	for (int i=0; i<DATA_N; i++) {
		h_Data[i] = 0;
	}

    // Annotate profiling state
    #ifdef GPU_PROFILING
    #endif

     //Get the NVIDIA platform
	ciErrNum = clGetPlatformIDs(1, &cpPlatform, NULL);

    //Get the devices
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &ciDeviceCount);
    cdDevices = (cl_device_id *)malloc(ciDeviceCount * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, ciDeviceCount, cdDevices, NULL);

    //Create the context
    cxGPUContext = clCreateContext(0, ciDeviceCount, cdDevices, NULL, NULL, &ciErrNum);

    // Set up command queue(s) for GPU's specified on the command line or all GPU's
	{
        // Find out how many GPU's to compute on all available GPUs
        size_t nDeviceBytes;
        ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
        ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

        for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
        {
            // get & log device index # and name
            deviceNr[i] = i;

            // cdDevice = oclGetDev(cxGPUContext, i);
			cdDevice = cdDevices[i];
			ciErrNum = clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cDeviceName), cDeviceName, NULL);

            // create a command que
            commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        }
    }

    // Load the OpenCL source code from the .cl file 
    char *source = "__kernel void reduce(__global float *d_Result, __global float *d_Input, int N){\n"\
		"const int tid = get_global_id(0);\n"\
		"const int threadN = get_global_size(0);\n"\
		"float sum = 0;\n"\
		"int count = 0;\n"\
		"for(int pos = tid; pos < N; pos += threadN) { \n"\
		"    sum += d_Input[pos];\n"\
		"	count++;\n"\
		"}\n"\
		"d_Result[tid] = sum / count;\n"\
		"}\n";

    // Create the program for all GPUs in the context
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, NULL, &ciErrNum);
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
		cerr << "could not build program" << endl;
    }

    // Create host buffer with page-locked memory
    h_DataBuffer = clCreateBuffer(cxGPUContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  DATA_N * sizeof(float), h_Data, &ciErrNum);

    // Create buffers for each GPU, with data divided evenly among GPU's
    int sizePerGPU = DATA_N / ciDeviceCount;
    int workOffset[MAX_GPU_COUNT];
    int workSize[MAX_GPU_COUNT];
    workOffset[0] = 0;
    for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
    {
        workSize[i] = (i != (ciDeviceCount - 1)) ? sizePerGPU : (DATA_N - workOffset[i]);        

        // Input buffer
        d_Data[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSize[i] * sizeof(float), NULL, &ciErrNum);

        // Copy data from host to device
        ciErrNum = clEnqueueCopyBuffer(commandQueue[i], h_DataBuffer, d_Data[i], workOffset[i] * sizeof(float), 
                                      0, workSize[i] * sizeof(float), 0, NULL, NULL);        

        // Output buffer
        d_Result[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, ACCUM_N * sizeof(float), NULL, &ciErrNum);
        
        // Create kernel
        reduceKernel[i] = clCreateKernel(cpProgram, "reduce", &ciErrNum);
        
        // Set the args values and check for errors
        ciErrNum |= clSetKernelArg(reduceKernel[i], 0, sizeof(cl_mem), &d_Result[i]);
        ciErrNum |= clSetKernelArg(reduceKernel[i], 1, sizeof(cl_mem), &d_Data[i]);
        ciErrNum |= clSetKernelArg(reduceKernel[i], 2, sizeof(int), &workSize[i]);

        workOffset[i + 1] = workOffset[i] + workSize[i];
    }

    // Set # of work items in work group and total in 1 dimensional range
    size_t localWorkSize[] = {THREAD_N};        
    size_t globalWorkSize[] = {ACCUM_N};        

    // Start timer and launch reduction kernel on each GPU, with data split between them 
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {        
        ciErrNum = clEnqueueNDRangeKernel(commandQueue[i], reduceKernel[i], 1, 0, globalWorkSize, localWorkSize,
                                         0, NULL, &GPUExecution[i]);
    }
    
    // Copy result from device to host for each device
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        ciErrNum = clEnqueueReadBuffer(commandQueue[i], d_Result[i], CL_FALSE, 0, ACCUM_N * sizeof(float), 
                            h_SumGPU + i *  ACCUM_N, 0, NULL, &GPUDone[i]);
    }

    // Synchronize with the GPUs and do accumulated error check
    clWaitForEvents(ciDeviceCount, GPUDone);

    // Aggregate results for multiple GPU's and stop/log processing time
    sumGPU = 0;
    for(unsigned int i = 0; i < ciDeviceCount * ACCUM_N; i++)
    {
         sumGPU += h_SumGPU[i];
    }

    // Print Execution Times for each GPU
    #ifdef GPU_PROFILING
        for(unsigned int i = 0; i < ciDeviceCount; i++) 
        {
        //    cdDevice = oclGetDev(cxGPUContext, deviceNr[i]);
        //    clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cDeviceName), cDeviceName, NULL);
        }
    #endif

    // Run the computation on the Host CPU and log processing time 
    sumCPU = 0;
    for(unsigned int i = 0; i < DATA_N; i++)
    {
        sumCPU += h_Data[i];
    }

    // Check GPU result against CPU result 
    dRelError = 100.0 * fabs(sumCPU - sumGPU) / fabs(sumCPU);

	cout << "sumGPU " << sumGPU << ", sumCPU " << sumCPU << endl;

    // cleanup 
    free(h_Data);
    for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
    {
        clReleaseKernel(reduceKernel[i]);
        clReleaseCommandQueue(commandQueue[i]);
    }
    clReleaseProgram(cpProgram);
    clReleaseContext(cxGPUContext);
	
	return 0;
  }
  */