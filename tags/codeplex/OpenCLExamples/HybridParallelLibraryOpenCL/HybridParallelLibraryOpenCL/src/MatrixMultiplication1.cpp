#include "../include/MatrixMultiplication1.h"

MatrixMultiplication1::MatrixMultiplication1() { }

MatrixMultiplication1::~MatrixMultiplication1() { }

std::string MatrixMultiplication1::getName() const {
	return "MatrixMultiplication1";
}

void MatrixMultiplication1::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
    double factor = 450.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
    this->sizeZ = (int) (sizeZ * factor);
}

void MatrixMultiplication1::performAlgorithm() {
	int error = 0;
	size_t dim = 0;
	size_t maxLocals = 0;
	size_t local[2];
	size_t global[2];

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
    error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxLocals, NULL);
    if (error != CL_SUCCESS)
        this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}

	local[0] = this->localWorkSize(maxLocals, this->sizeX, 2);
	local[1] = this->localWorkSize(maxLocals, this->sizeY, 2);
	global[0] = this->globalWorkSize(local[0], this->sizeX);
	global[1] = this->globalWorkSize(local[1], this->sizeY);

	enqueueAndFinish(this->kernel, this->commands, global, local, 2);
}

const char *MatrixMultiplication1::algorithm() const {
	return "																				\n "\
		"__kernel void algorithm(															\n "\
		"	__global float *inputMatrixA,													\n "\
		"	__global float *inputMatrixB,													\n "\
		"	__global float *resultMatrixC,													\n "\
		"	__global size_t *sizeX,															\n "\
		"	__global size_t *sizeY,															\n "\
		"	__global size_t *sizeZ)															\n "\
		"{																					\n "\
		"	size_t idX = get_global_id(0);													\n "\
		"	size_t idY = get_global_id(1);													\n "\
		"																					\n "\
		"	if(idX < *sizeX && idY < *sizeY)												\n "\
		"	{																				\n "\
		"		float v = 0;																\n "\
		"																					\n "\
		"		for(int k = 0; k < *sizeZ; k++)												\n "\
		"		{																			\n "\
		"			v += inputMatrixA[idX * *sizeX + k] * inputMatrixB[k * *sizeZ + idY];	\n "\
		"		}																			\n "\
		"																					\n "\
		"		resultMatrixC[idX * *sizeX + idY] = v;										\n "\
		"	}																				\n "\
		"}																					";
}