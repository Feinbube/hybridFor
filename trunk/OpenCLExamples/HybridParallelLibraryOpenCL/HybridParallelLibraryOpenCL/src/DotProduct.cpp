#include "../include/DotProduct.h"

const float DotProduct::validationThreshold = 0.99999f;

DotProduct::DotProduct()
	:	matrixA(NULL),
		matrixB(NULL),
		memSizeX(NULL),
		memSizeY(NULL),
		tempResults(NULL),
		arrayMatrixA(NULL),
		arrayMatrixB(NULL),
		result(0) { }

DotProduct::~DotProduct() {
	this->discardMembers();
}

std::string DotProduct::getName() const {
	return "DotProduct";
}

void DotProduct::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
	float factor = 300.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
}

void DotProduct::discardMembers() {
    if (this->matrixA) clReleaseMemObject(this->matrixA);
    if (this->matrixB) clReleaseMemObject(this->matrixB);
    if (this->memSizeX) clReleaseMemObject(this->memSizeX);
    if (this->memSizeY) clReleaseMemObject(this->memSizeY);
    if (this->tempResults) clReleaseMemObject(this->tempResults);
	if (this->arrayMatrixA) delete arrayMatrixA;
	if (this->arrayMatrixB) delete arrayMatrixB;
}

void DotProduct::initializeMembers() {
	int error;

    this->matrixA = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeX * this->sizeY, NULL, NULL);
    this->matrixB = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeX * this->sizeY, NULL, NULL);
    this->tempResults = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeX, NULL, NULL);
    this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->memSizeY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
	this->arrayMatrixA = new size_t[this->sizeX * this->sizeY];
	this->arrayMatrixB = new size_t[this->sizeX * this->sizeY];

    for (int i = 0; i < this->sizeX * this->sizeY; i++) {
        this->arrayMatrixA[i] = i;
        this->arrayMatrixB[i] = i * 2;
    }

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clEnqueueWriteBuffer(this->commands[i], this->matrixA, CL_TRUE, 0, sizeof(size_t) * this->sizeX * this->sizeY, this->arrayMatrixA, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->matrixB, CL_TRUE, 0, sizeof(size_t) * this->sizeX * this->sizeY, this->arrayMatrixB, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeY, CL_TRUE, 0, sizeof(size_t), &this->sizeY, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void DotProduct::performAlgorithm() {
	int error = 0;
	size_t local = 0;
	size_t global = 0;
	size_t *tempResultsArray = new size_t[this->sizeX];

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->matrixA);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->matrixB);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->memSizeY);
		error |= clSetKernelArg(this->kernel[i], 4, sizeof(cl_mem), &this->tempResults);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}
	global = this->globalWorkSize(local, this->sizeX);
	
	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueNDRangeKernel(this->commands[i], this->kernel[i], 1, NULL, &global, &local, 0, NULL, NULL);
		if (error)
			this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
	    clFinish(this->commands[i]);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueReadBuffer(this->commands[i], this->tempResults, CL_TRUE, 0, sizeof(size_t) * this->sizeX, tempResultsArray, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);
	}

	this->result = 0;
    for (int index = 0; index < this->sizeX; index++)
        this->result += tempResultsArray[index];
}

const char *DotProduct::algorithm() const {
	return "														\n "\
		"__kernel void algorithm(									\n "\
		"	__global size_t *matrixA,								\n "\
		"	__global size_t *matrixB,								\n "\
		"	__global size_t *sizeX,									\n "\
		"	__global size_t *sizeY,									\n "\
		"	__global size_t *tempResults)							\n "\
		"{															\n "\
		"	size_t x = get_global_id(0);							\n "\
		"															\n "\
		"	if (x < *sizeX) {										\n "\
		"		tempResults[x] = 0.0f;								\n "\
		"		size_t tid = x;										\n "\
		"															\n "\
		"		for (int y = 0; y < *sizeY; y++) {					\n "\
		"			tempResults[x] += matrixA[tid] * matrixB[tid];	\n "\
		"			tid += *sizeX;									\n "\
		"		}													\n "\
		"	}														\n "\
		"}															";
}

const bool DotProduct::isValid() const {
	size_t validationResult = 0;

	for (int i = 0; i < this->sizeX * this->sizeY; i++)
		validationResult += this->arrayMatrixA[i] * this->arrayMatrixB[i];

	if (this->result / validationResult < DotProduct::validationThreshold)
		return false;
	else
		return true;
}
