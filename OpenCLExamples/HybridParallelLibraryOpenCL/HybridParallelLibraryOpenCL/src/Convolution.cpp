#include "../include/Convolution.h"

const float Convolution::validationThreshold = 0.00005f;

Convolution::Convolution()
	:	startImage(NULL),
		outImage(NULL),
		memSizeX(NULL),
		memSizeY(NULL) { }

Convolution::~Convolution() {
	this->discardMembers();
}

std::string Convolution::getName() const {
	return "Convolution";
}

void Convolution::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
    double factor = 2000.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
    this->sizeZ = (int) (sizeZ * factor);
}

void Convolution::discardMembers() {
    if (this->startImage) clReleaseMemObject(this->startImage);
    if (this->outImage) clReleaseMemObject(this->outImage);
    if (this->memSizeX) clReleaseMemObject(this->memSizeX);
    if (this->memSizeY) clReleaseMemObject(this->memSizeY);
}

void Convolution::initializeMembers() {
	int error = 0;

    this->startImage = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * this->sizeX * this->sizeY, NULL, NULL);
    this->outImage = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * this->sizeX * this->sizeY, NULL, NULL);
    this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->memSizeY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

    float *startImageArray = new float[this->sizeX * this->sizeY];
    float *outImageArray = new float[this->sizeX * this->sizeY];
	this->fillInData(startImageArray, outImageArray);
	
	for (cl_uint i=0; i<this->deviceCount; i++) {
		error  = clEnqueueWriteBuffer(this->commands[i], this->startImage, CL_TRUE, 0, sizeof(float) * this->sizeX * this->sizeY, startImageArray, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->outImage, CL_TRUE, 0, sizeof(float) * this->sizeX * this->sizeY, outImageArray, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeY, CL_TRUE, 0, sizeof(size_t), &this->sizeY, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}

	delete startImageArray;
	delete outImageArray;
}

void Convolution::fillInData(float *&startImageArray, float *&outImageArray) const {  
	float EDGE = 5.0f;

    for (int x = 0; x < this->sizeX; x++)
        for (int y = 0; y < this->sizeY; y++)
            startImageArray[x * this->sizeX + y] = this->randomFloat() * 10;

    for (int x = 0; x < this->sizeX; x++)
        for (int y = 0; y < this->sizeY; y++)
            outImageArray[x * this->sizeX + y] = 0.0f;

    for (int x = 0; x < this->sizeX; x++) 	{
		startImageArray[x] = outImageArray[x] = EDGE;
		startImageArray[(this->sizeY - 1) * this->sizeX + x] = outImageArray[(this->sizeY - 1) * this->sizeX + x] = EDGE;
	}

    for (int y = 0; y < this->sizeY; y++) 	{
		startImageArray[y * this->sizeX] = outImageArray[y * this->sizeX] = EDGE;
		startImageArray[(y + 1) * this->sizeX - 1] = outImageArray[(y + 1) * this->sizeX - 1] = EDGE;
	}
}

void Convolution::performAlgorithm() {
	int error = 0;
	size_t dim = 0;
	size_t maxLocals = 0;
	size_t local[2];
	size_t global[2];

	for (cl_uint i=0; i<this->deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->startImage);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->outImage);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->memSizeY);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (cl_uint i=0; i<this->deviceCount; i++) {
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

const char *Convolution::algorithm() const {
	return "																	\n "\
		"__kernel void algorithm(												\n "\
		"	__global float *startImage,											\n "\
		"	__global float *outImage,											\n "\
		"	__global size_t *sizeX,												\n "\
		"	__global size_t *sizeY)												\n "\
		"{																		\n "\
		"	size_t idx = get_global_id(0);										\n "\
		"	size_t idy = get_global_id(1);										\n "\
		"																		\n "\
		"	if(idx < (*sizeX - 1) && idx > 0 && idy < (*sizeY - 1) && idy > 0)	\n "\
		"	{																	\n "\
		"		outImage[idx * *sizeX + idy] = 0.2f * (							\n "\
		"				startImage[idx * *sizeX + idy]							\n "\
		"			+	startImage[idx * *sizeX + idy + 1]						\n "\
		"			+	startImage[idx * *sizeX + idy - 1]						\n "\
		"			+	startImage[(idx + 1) * *sizeX + idy]					\n "\
		"			+	startImage[(idx - 1) * *sizeX + idy]);					\n "\
		"	}																	\n "\
		"}																		";
}

// TODO Test & Fix
const bool Convolution::isValid() const {
	int error;
	bool result;

    float *startImageArray = new float[this->sizeX * this->sizeY];
    float *outImageArray = new float[this->sizeX * this->sizeY];

    error = clEnqueueReadBuffer(this->commands[0], this->startImage, CL_TRUE, 0, sizeof(float) * this->sizeX * this->sizeY, startImageArray, 0, NULL, NULL);
    error |= clEnqueueReadBuffer(this->commands[0], this->outImage, CL_TRUE, 0, sizeof(float) * this->sizeX * this->sizeY, outImageArray, 0, NULL, NULL);
    if (error != CL_SUCCESS)
		this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);

	result = this->validateResult(startImageArray, outImageArray);

	delete startImageArray;
	delete outImageArray;

	return result;
}

const bool Convolution::validateResult(float const* const& startImageArray, float const* const& outImageArray) const {
    for (int x = 1; x < this->sizeX - 1; x++)
        for (int y = 1; y < this->sizeY - 1; y++)
            if(		outImageArray[x * this->sizeX + y]
				-	0.2f * (
						startImageArray[x * this->sizeX + y]
					+	startImageArray[x * this->sizeX + y + 1]
					+	startImageArray[x * this->sizeX + y - 1]
					+	startImageArray[(x + 1) * this->sizeX + y]
					+	startImageArray[(x - 1) * this->sizeX + y])
				> Convolution::validationThreshold)
			{
                return false;
			}

    return true;
}