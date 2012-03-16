#include "../include/MatrixMultiplication3.h"

MatrixMultiplication3::MatrixMultiplication3() { }

MatrixMultiplication3::~MatrixMultiplication3() { }

std::string MatrixMultiplication3::getName() const {
	return "MatrixMultiplication3";
}

void MatrixMultiplication3::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
    double factor = 68.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
    this->sizeZ = (int) (sizeZ * factor);
}

void MatrixMultiplication3::performAlgorithm() {
	int error = 0;
	size_t local = 0;
	size_t global = 0;

	for (unsigned int i=0; i<this->deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->matrixA);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->matrixB);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->matrixC);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel[i], 4, sizeof(cl_mem), &this->memSizeY);
		error |= clSetKernelArg(this->kernel[i], 5, sizeof(cl_mem), &this->memSizeZ);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<this->deviceCount; i++) { 
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}

	global = this->globalWorkSize(local, this->sizeX);
	
	enqueueAndFinish(this->kernel, this->commands, &global, &local, 1);
}

const char *MatrixMultiplication3::algorithm() const {
	return "																						\n "\
		"__kernel void algorithm(																	\n "\
		"	__global float *inputMatrixA,															\n "\
		"	__global float *inputMatrixB,															\n "\
		"	__global float *resultMatrixC,															\n "\
		"	__global size_t *sizeX,																	\n "\
		"	__global size_t *sizeY,																	\n "\
		"	__global size_t *sizeZ)																	\n "\
		"{																							\n "\
		"	int ITILE2 = 32;																		\n "\
		"	int JTILE2 = 32;																		\n "\
		"	size_t id = get_global_id(0);															\n "\
		"																							\n "\
		"	if(id < (*sizeX + ITILE2 - 1) / ITILE2)													\n "\
		"	{																						\n "\
		"		int ii = id * ITILE2;																\n "\
		"		int il = ii + ITILE2 < *sizeX ? ii + ITILE2 : *sizeX;								\n "\
		"																							\n "\
		"		for(int i = ii; i < il; i++)														\n "\
		"			for(int j = 0; j < *sizeY; j++)													\n "\
		"				resultMatrixC[i * *sizeX + j] = 0;											\n "\
		"																							\n "\
		"		for(int jj = 0; jj < *sizeY; jj += JTILE2)											\n "\
		"		{																					\n "\
		"			int jl = jj + JTILE2 < *sizeY ? jj + JTILE2 : *sizeY;							\n "\
		"																							\n "\
		"			for(int i = ii; i < il; i++)													\n "\
		"				for(int j = jj; j < jl; j++)												\n "\
		"					for(int k = 0; k < *sizeZ; k++)											\n "\
		"						resultMatrixC[i * *sizeX + j] +=									\n "\
		"							inputMatrixA[i * *sizeX + k] * inputMatrixB[k * *sizeZ + j];	\n "\
		"		}																					\n "\
		"	}																						\n "\
		"}																							";
}
