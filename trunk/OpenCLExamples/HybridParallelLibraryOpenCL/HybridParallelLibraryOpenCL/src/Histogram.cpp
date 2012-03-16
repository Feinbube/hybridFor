#include "../include/Histogram.h"

Histogram::Histogram()
	:	temp(NULL),
		buffer(NULL),
		histo(NULL),
		memSizeX(NULL),
		memSizeY(NULL),
		memSizeZ(NULL),
		program2(NULL),
		program3(NULL) { }

Histogram::~Histogram() {
	this->discardMembers();
}

std::string Histogram::getName() const {
	return "Histogram";
}

void Histogram::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
    float factor = 5000000.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
    this->sizeZ = 256;
}

void Histogram::discardMembers() {
    if (this->temp) clReleaseMemObject(this->temp);
    if (this->buffer) clReleaseMemObject(this->buffer);
    if (this->histo) clReleaseMemObject(this->histo);
    if (this->memSizeX) clReleaseMemObject(this->memSizeX);
    if (this->memSizeY) clReleaseMemObject(this->memSizeY);
    if (this->memSizeZ) clReleaseMemObject(this->memSizeZ);
    if (this->program2) clReleaseProgram(this->program2);
    if (this->program3) clReleaseProgram(this->program3);

	releaseKernels(this->kernel2);
	releaseKernels(this->kernel3);
}

void Histogram::initializeMembers() {
	int error;

    this->temp = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeZ, NULL, NULL);
    this->buffer = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeX, NULL, NULL);
    this->histo = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeZ, NULL, NULL);
    this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->memSizeY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->memSizeZ = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
	this->bufferArray = new size_t[this->sizeX];
	size_t *tempArray = new size_t[this->sizeZ];
	size_t *histoArray = new size_t[this->sizeZ];

    for (int x = 0; x < sizeX; x++)
        this->bufferArray[x] = (size_t) rand() % this->sizeZ;
    for (int z = 0; z < sizeZ; z++)
        tempArray[z] = histoArray[z] = 0;
	
	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clEnqueueWriteBuffer(this->commands[i], this->buffer, CL_TRUE, 0, sizeof(size_t) * this->sizeX, this->bufferArray, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->temp, CL_TRUE, 0, sizeof(size_t) * this->sizeZ, tempArray, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->histo, CL_TRUE, 0, sizeof(size_t) * this->sizeZ, histoArray, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeY, CL_TRUE, 0, sizeof(size_t), &this->sizeY, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeZ, CL_TRUE, 0, sizeof(size_t), &this->sizeZ, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}

	delete histoArray;
}

void Histogram::performAlgorithm() {
	int error = 0;
	size_t local = 0;
	size_t globalY = 0;
	size_t globalZ = 0;

	for (unsigned int i=0; i<deviceCount; i++) {
		// TODO muss das auch mit kernel2 und kernel3 passieren?
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}

	globalY = this->globalWorkSize(local, this->sizeY);
	globalZ = this->globalWorkSize(local, this->sizeZ);

	this->executeAlgorithm(globalZ, local);
	this->executeAlgorithm2(globalY, local);
	this->executeAlgorithm3(globalZ, local);
}

void Histogram::executeAlgorithm(size_t global, size_t local) {
	int error;

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->temp);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->memSizeZ);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	enqueueAndFinish(this->kernel, this->commands, &global, &local, 1);
}

void Histogram::executeAlgorithm2(size_t global, size_t local) {
	int error;

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel2[i], 0, sizeof(cl_mem), &this->temp);
		error |= clSetKernelArg(this->kernel2[i], 1, sizeof(cl_mem), &this->buffer);
		error |= clSetKernelArg(this->kernel2[i], 2, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel2[i], 3, sizeof(cl_mem), &this->memSizeY);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	enqueueAndFinish(this->kernel2, this->commands, &global, &local, 1);
}

void Histogram::executeAlgorithm3(size_t global, size_t local) {
	int error;

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel3[i], 0, sizeof(cl_mem), &this->temp);
		error |= clSetKernelArg(this->kernel3[i], 1, sizeof(cl_mem), &this->histo);
		error |= clSetKernelArg(this->kernel3[i], 2, sizeof(cl_mem), &this->memSizeZ);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	enqueueAndFinish(this->kernel3, this->commands, &global, &local, 1);
}

const char *Histogram::algorithm() const {
	return "								\n "\
		"__kernel void algorithm(			\n "\
		"	__global size_t *temp,			\n "\
		"	__global size_t *sizeZ)			\n "\
		"{									\n "\
		"	size_t z = get_global_id(0);	\n "\
		"									\n "\
		"	if (z < *sizeZ)					\n "\
		"	{								\n "\
		"		temp[z] = 0;				\n "\
		"	}								\n "\
		"}									";
		
}

const char *Histogram::algorithm2() const {
	return "																\n "\
		"#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable	\n" \
		"																	\n "\
		"__kernel void algorithm2(											\n "\
		"	__global size_t *temp,											\n "\
		"	__global size_t *buffer,										\n "\
		"	__global size_t *sizeX,											\n "\
		"	__global size_t *sizeY)											\n "\
		"{																	\n "\
		"	size_t y = get_global_id(0);									\n "\
		"																	\n "\
		"	if (y < *sizeX)													\n "\
		"		while (y < *sizeX) {										\n "\
		"			atom_inc(&temp[buffer[y]]); 							\n "\
		"			y += *sizeY;											\n "\
		"		}															\n "\
		"}																	";
}

const char *Histogram::algorithm3() const {
	return "																\n "\
		"#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable	\n" \
		"																	\n "\
		"__kernel void algorithm3(											\n "\
		"	__global size_t *temp,											\n "\
		"	__global size_t *histo,											\n "\
		"	__global size_t *sizeZ)											\n "\
		"{																	\n "\
		"	size_t z = get_global_id(0);									\n "\
		"																	\n "\
		"	if (z < *sizeZ)													\n "\
		"		atom_add(&histo[z], temp[z]);								\n "\
		"}																		";
}

void Histogram::initializeOpenCL() {
    int err;

	ExampleBase::initializeOpenCL();

	if (!this->supportsBaseAtomics())
        this->exitWith("Error: No support for base atomic operations for " + this->getName() + "!", 0);
	
	const char
		*algo2 = this->algorithm2(),
		*algo3 = this->algorithm3();
	this->program2 = clCreateProgramWithSource(this->context, 1, (const char **) &algo2, NULL, &err);
	this->program3 = clCreateProgramWithSource(this->context, 1, (const char **) &algo3, NULL, &err);
	if (!this->program2 || !this->program3)
        this->exitWith("Error: Failed to create compute program for " + this->getName() + "!", err);

    err  = clBuildProgram(this->program2, 0, NULL, NULL, NULL, NULL);
    err |= clBuildProgram(this->program3, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)     {
        size_t len;
        char buffer[2048];

        clGetProgramBuildInfo(this->program2, this->device_ids[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        clGetProgramBuildInfo(this->program3, this->device_ids[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        this->exitWith("Error: Failed to build program executable for " + this->getName() + "!", err);
    }

	createKernels(this->kernel2, this->program2);
	createKernels(this->kernel3, this->program3);
}

const bool Histogram::isValid() const {
	bool result = true;
	size_t *histoArray = new size_t[this->sizeZ];

    int error = clEnqueueReadBuffer(this->commands[0], this->histo, CL_TRUE, 0, sizeof(size_t) * this->sizeZ, histoArray, 0, NULL, NULL);
    if (error != CL_SUCCESS)
		this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);

    for (int x = 0; x < this->sizeX; x++)
		histoArray[this->bufferArray[x]]--;

    for (int z = 0; z < this->sizeZ; z++)
        if (histoArray[z] != 0) {
			cout << z << " | " << histoArray[z] << endl;
			result = false;
			break;
		}

	delete histoArray;
    return result;
}
