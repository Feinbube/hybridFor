#include "../include/PrefixScan.h"

PrefixScan::PrefixScan()
	:	startData(NULL),
	scannedData(NULL),
	memSizeX(NULL),
	startDataArray(NULL),
	scannedDataArray(NULL),
	program2(NULL) { }

PrefixScan::~PrefixScan() {
	this->discardMembers();
}

std::string PrefixScan::getName() const {
	return "PrefixScan";
}

void PrefixScan::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
	this->sizeX = (int) (sizeX * 500.0);
}

void PrefixScan::discardMembers() {
	if (this->startData) clReleaseMemObject(this->startData);
	if (this->scannedData) clReleaseMemObject(this->scannedData);
	if (this->memSizeX) clReleaseMemObject(this->memSizeX);
	if (this->startDataArray) delete this->startDataArray;
	if (this->scannedDataArray) delete this->scannedDataArray;
	if (this->program2) clReleaseProgram(this->program2);

	for (unsigned int i=0; i<deviceCount; i++) {
	//	if (this->kernel2[i])
	//		clReleaseKernel(this->kernel2[i]);
	}
}

void PrefixScan::initializeMembers() {
	int error = 0;

	this->startData = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * this->sizeX, NULL, NULL);
	this->scannedData = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * this->sizeX, NULL, NULL);
	this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

	this->startDataArray = new float[this->sizeX];
	this->scannedDataArray = new float[this->sizeX];
	for (int x = 0; x < this->sizeX; x++)
		this->startDataArray[x] = this->randomFloat();

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueWriteBuffer(this->commands[i], this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void PrefixScan::performAlgorithm() {
	int error;
	size_t local, global, increment = 1;
	float *startDataCopy, *data1, *data2, *tmpData;
	cl_mem incrementMemory = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}
	global = this->globalWorkSize(local, this->sizeX);

	this->scannedDataArray[0] = this->startDataArray[0];
	startDataCopy = new float[this->sizeX];
	for (int x = 0; x < this->sizeX; x++)
		startDataCopy[x] = this->startDataArray[x];
	data1 = startDataCopy;
	data2 = this->scannedDataArray;

	this->setKernelArguments(incrementMemory);

	while (increment < (size_t) this->sizeX) {
		for (unsigned int i=0; i<deviceCount; i++) {
			// nur teildaten auf gpu kopieren, startindex für gpu übergeben
			error  = clEnqueueWriteBuffer(this->commands[i], this->startData, CL_TRUE, 0, sizeof(float) * this->sizeX, data1, 0, NULL, NULL);
			error |= clEnqueueWriteBuffer(this->commands[i], this->scannedData, CL_TRUE, 0, sizeof(float) * this->sizeX, data2, 0, NULL, NULL);
			error |= clEnqueueWriteBuffer(this->commands[i], incrementMemory, CL_TRUE, 0, sizeof(size_t), &increment, 0, NULL, NULL);
			if (error != CL_SUCCESS)
				this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
		}

		this->executeAlgorithm(global, local);
		this->executeAlgorithm2(global, local);

		for (unsigned int i=0; i<deviceCount; i++) {
			// 
			error = clEnqueueReadBuffer(this->commands[i], this->scannedData, CL_TRUE, 0, sizeof(float) * this->sizeX, data2, 0, NULL, NULL);
			if (error != CL_SUCCESS)
				this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);
		}

		increment <<= 1;
		tmpData = data1;
		data1 = data2;
		data2 = tmpData;
	}

	delete startDataCopy;
	clReleaseMemObject(incrementMemory);
}

void PrefixScan::setKernelArguments(cl_mem &incrementMemory) const {
	int error;

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->startData);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->scannedData);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &incrementMemory);
		error |= clSetKernelArg(this->kernel2[i], 0, sizeof(cl_mem), &this->startData);
		error |= clSetKernelArg(this->kernel2[i], 1, sizeof(cl_mem), &this->scannedData);
		error |= clSetKernelArg(this->kernel2[i], 2, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel2[i], 3, sizeof(cl_mem), &incrementMemory);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}
}

void PrefixScan::executeAlgorithm(size_t global, size_t local) {
	int error;
	
	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueNDRangeKernel(this->commands[i], this->kernel[i], 1, NULL, &global, &local, 0, NULL, NULL);
		if (error)
			this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		clFinish(this->commands[i]);
	}
}

void PrefixScan::executeAlgorithm2(size_t global, size_t local) {
	int error;

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueNDRangeKernel(this->commands[i], this->kernel2[i], 1, NULL, &global, &local, 0, NULL, NULL);
		if (error)
			this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		clFinish(this->commands[i]);
	}
}

const char *PrefixScan::algorithm() const {
	return "											\n "\
		"__kernel void algorithm(						\n "\
		"	__global float *data1,						\n "\
		"	__global float *data2,						\n "\
		"	__global size_t *sizeX,						\n "\
		"	__global size_t *increment)					\n "\
		"{												\n "\
		"	size_t x = get_global_id(0);				\n "\
		"												\n "\
		"	if (x < *sizeX && x < *increment && x > 0)	\n "\
		"			data2[x] = data1[x];				\n "\
		"}												";
}

const char *PrefixScan::algorithm2() const {
	return "													\n "\
		"__kernel void algorithm2(								\n "\
		"	__global float *data1,								\n "\
		"	__global float *data2,								\n "\
		"	__global size_t *sizeX,								\n "\
		"	__global size_t *increment)							\n "\
		"{														\n "\
		"	size_t x = get_global_id(0);						\n "\
		"														\n "\
		"	if (x < *sizeX && x >= *increment)					\n "\
		"		data2[x] = data1[x] + data1[x - *increment];	\n "\
		"}															";
}

void PrefixScan::initializeOpenCL() {
	int err;

	ExampleBase::initializeOpenCL();

	const char *algo2 = this->algorithm2();
	this->program2 = clCreateProgramWithSource(this->context, 1, (const char **) &algo2, NULL, &err);
	if (!this->program2)
		this->exitWith("Error: Failed to create compute program for " + this->getName() + "!", err);

	err = clBuildProgram(this->program2, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)     {
		size_t len;
		char buffer[2048];

		clGetProgramBuildInfo(this->program2, this->device_ids[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		this->exitWith("Error: Failed to build program executable for " + this->getName() + "!", err);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		this->kernel2[i] = clCreateKernel(this->program2, "algorithm2", &err);
		if (!this->kernel2[i] || err != CL_SUCCESS)
			this->exitWith("Error: Failed to create compute kernel for " + this->getName() + "!", err);
	}
}

// TODO was bedeutet das "Fix me"? Ging das jemals?
// Fix me?
const bool PrefixScan::isValid() const {
	int error;

	// TODO iterate across multiple GPU command queues 
	error = clEnqueueReadBuffer(this->commands[0], this->scannedData, CL_TRUE, 0, sizeof(float) * this->sizeX, this->scannedDataArray, 0, NULL, NULL);
	if (error != CL_SUCCESS)
		this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);

	for (int x = 0; x < this->sizeX / 2; x++) {
		if (this->scannedDataArray[x] > this->scannedDataArray[x + 1])
			return false;
	}

	return true;
}
