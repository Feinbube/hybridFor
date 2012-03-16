#include "../include/MatrixMultiplication4.h"

MatrixMultiplication4::MatrixMultiplication4() { }

MatrixMultiplication4::~MatrixMultiplication4() { }

std::string MatrixMultiplication4::getName() const {
	return "MatrixMultiplication4";
}

void MatrixMultiplication4::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
    double factor = 75.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
    this->sizeZ = (int) (sizeZ * factor);
}

void MatrixMultiplication4::performAlgorithm() {
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

const char *MatrixMultiplication4::algorithm() const {
	return "																						\n "\
		"__kernel void algorithm(																	\n "\
		"	__global float *inputMatrixA,															\n "\
		"	__global float *inputMatrixB,															\n "\
		"	__global float *resultMatrixC,															\n "\
		"	__global size_t *sizeX,																	\n "\
		"	__global size_t *sizeY,																	\n "\
		"	__global size_t *sizeZ)																	\n "\
		"{																							\n "\
		"	int ITILE3 = 32;																		\n "\
		"	int JTILE3 = 32;																		\n "\
		"	int KTILE3 = 32;																		\n "\
		"																							\n "\
		"	size_t idx = get_global_id(0);															\n "\
		"	size_t idy = get_global_id(1);															\n "\
		"																							\n "\
		"	if(		idx < (*sizeX + ITILE3 - 1) / ITILE3											\n "\
		"		&&	idy < (*sizeY + JTILE3 - 1) / JTILE3)											\n "\
		"	{																						\n "\
		"		int ii = idx * ITILE3;																\n "\
		"		int jj = idy * JTILE3;																\n "\
		"		int il = ii + ITILE3 < *sizeX ? ii + ITILE3 : *sizeX;								\n "\
		"		int jl = jj + JTILE3 < *sizeY ? jj + JTILE3 : *sizeY;								\n "\
		"																							\n "\
		"		for(int i = ii; i < il; i++)														\n "\
		"			for(int j = jj; j < jl; j++)													\n "\
		"				resultMatrixC[i * *sizeX + j] = 0;											\n "\
		"																							\n "\
		"		for(int kk = 0; kk < *sizeZ; kk += KTILE3)											\n "\
		"		{																					\n "\
		"			int kl = kk + KTILE3 < *sizeZ ? kk + KTILE3 : *sizeZ;							\n "\
		"																							\n "\
		"			for(int i = ii; i < il; i++)													\n "\
		"				for(int j = jj; j < jl; j++)												\n "\
		"					for(int k = kk; k < kl; k++)											\n "\
		"						resultMatrixC[i * *sizeX + j] +=									\n "\
		"							inputMatrixA[i * *sizeX + k] * inputMatrixB[k * *sizeZ + j];	\n "\
		"		}																					\n "\
		"	}																						\n "\
		"}																							";
}
