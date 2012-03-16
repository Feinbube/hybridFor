#include "../include/MatrixMultiplication0.h"

MatrixMultiplication0::MatrixMultiplication0() { }

MatrixMultiplication0::~MatrixMultiplication0() { }

std::string MatrixMultiplication0::getName() const {
	return "MatrixMultiplication0";
}

void MatrixMultiplication0::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
#ifdef DEBUG
    this->sizeX = 4;
    this->sizeY = 4;
    this->sizeZ = 4;
#else
    const double factor = 275.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
    this->sizeZ = (int) (sizeZ * factor);
#endif
}

void MatrixMultiplication0::performAlgorithm() {
	int error = 0;
	size_t local = 0;
	size_t global = 0;

	for (unsigned int i=0; i<this->deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->matrixA[i]);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->matrixB);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->matrixC[i]);
		// error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->memSizeX[i]);
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
#ifdef DEBUG
	cout << "global(" << global << ") local(" << local << ")" << endl; 
#endif
	enqueueAndFinish(this->kernel, this->commands, &global, &local, 1);
}

const char *MatrixMultiplication0::algorithm() const {
	return "																				\n "\
		"__kernel void algorithm(									\n"\
		"	__global float *inputMatrixA,								\n"\
		"	__global float *inputMatrixB,								\n"\
		"	__global float *resultMatrixC,								\n"\
		"	__global size_t *sizeX,									\n"\
		"	__global size_t *sizeY,									\n"\
		"	__global size_t *sizeZ)									\n"\
		"{												\n"\
		"	size_t id = get_global_id(0);								\n"\
		"												\n"\
		"	if(id < *sizeX)										\n"\
		"	{											\n"\
		"		for(int j = 0; j < *sizeY; j++)							\n"\
		"		{										\n"\
		"			float v = 0;								\n"\
		"												\n"\
		"			for(int k = 0; k < *sizeZ; k++)						\n"\
		"			{									\n"\
		"				v += inputMatrixA[id * *sizeX + k] * inputMatrixB[k * *sizeZ + j];	\n"\
		"			}									\n"\
		"												\n"\
		"			resultMatrixC[id * *sizeX + j] = v;					\n"\
		"		}										\n"\
		"	}											\n"\
		"}												";
}
