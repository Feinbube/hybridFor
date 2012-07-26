#ifndef EXAMPLEBASE
#define EXAMPLEBASE

#include <string>
#include <iostream>
#include <time.h>
#include <limits>
#include <CL/cl.h>
#include <math.h>
#include <string.h>
#include <float.h> // for FLT_MAX
#include <stdio.h>
#include <cstdlib>

using namespace std;

const unsigned int MAX_GPU_COUNT = 8;

class ExampleBase {
	public:
		ExampleBase();
		~ExampleBase();
		virtual std::string getName() const = 0;
		void Run(float sizeX, float sizeY, float sizeZ, int rounds, int warmupRounds);

	protected:
		cl_platform_id platform_id;
		cl_context context;
		cl_program program;
		// variables for multiple GPUs / accelerators
		cl_uint deviceCount;
		cl_device_id * device_ids;
		cl_command_queue commands[MAX_GPU_COUNT];
		cl_kernel kernel[MAX_GPU_COUNT];

		int sizePerDevice;
		int workSize[MAX_GPU_COUNT];
		int workOffset[MAX_GPU_COUNT];
	
		int sizeX;
		int sizeY;
		int sizeZ;

		const float randomFloat() const;
		virtual const bool isValid() const = 0;
		virtual void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) = 0;
		void setup();
		virtual void discardMembers() = 0;
		virtual void initializeMembers() = 0;
		virtual void performAlgorithm() = 0;
		const bool supportsBaseAtomics() const;
		const size_t localWorkSize(const size_t maxLocalWorkSize, const int numberOfWorkItems, const int dimensions) const;
		const size_t globalWorkSize(const size_t localWorkSize, const int numberOfWorkItems) const;
		virtual const char *algorithm() const = 0;
		virtual void initializeOpenCL();
		void exitWith(string message, int error) const;
		void printField(float const* const& field, int maxX) const;
		void printField(float const* const& field, int maxX, int maxY) const;

		inline const int indexOf(const int x, const int y) const {
			return x * this->sizeX + y;
		}

		void releaseKernels(cl_kernel * kernels) {
			for (unsigned int i=0; i<this->deviceCount; i++) {
				if (kernel[i])
					clReleaseKernel(kernel[i]);
			}
		}

		void createKernels(cl_kernel * kernels, cl_program) {
			for (unsigned int i=0; i<this->deviceCount; i++) {
				int err;
				kernels[i] = clCreateKernel(program, "algorithm", &err);

				if (!kernels[i] || err != CL_SUCCESS) {
					this->exitWith("Error: Failed to create compute kernel for " + this->getName() + "!", err);
				}
			}	
		}

		void enqueueAndFinish(cl_kernel * kernels, cl_command_queue * commands, size_t * global, size_t * local, const cl_uint workDim) {
			int error;	

			for (unsigned int i=0; i<this->deviceCount; i++) {
				error = clEnqueueNDRangeKernel(commands[i], kernels[i], workDim, NULL, global, local, 0, NULL, NULL);
				if (error)
					this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
			}

			for (unsigned int i=0; i<this->deviceCount; i++) {
				clFinish(commands[i]);
			}
		}
		void releaseMemObjects(cl_mem  objects[MAX_GPU_COUNT]) {
			for (unsigned int i=0; i<MAX_GPU_COUNT; i++) { 
				releaseMemObject(objects[i]);
			}	
		}
		void releaseMemObject(cl_mem & object) {
			if (object) {
				clReleaseMemObject(object);
			}
		}
	private:

		void runTimes(int numberOfTimes);
};

#endif
