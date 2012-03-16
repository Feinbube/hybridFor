#include "../include/HeatTransfer.h"

HeatTransfer::HeatTransfer()
	:	input(NULL),
		bitmap(NULL),
		memSizeX(NULL),
		memSizeY(NULL) { }

HeatTransfer::~HeatTransfer() {
	this->discardMembers();
}

std::string HeatTransfer::getName() const {
	return "HeatTransfer";
}

void HeatTransfer::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
	float factor = 200.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
	this->sizeZ = this->sizeX / 10;
}

void HeatTransfer::discardMembers() {
    if (this->input) clReleaseMemObject(this->input);
    if (this->bitmap) clReleaseMemObject(this->bitmap);
    if (this->memSizeX) clReleaseMemObject(this->memSizeX);
    if (this->memSizeY) clReleaseMemObject(this->memSizeY);
}

void HeatTransfer::initializeMembers() {
	int error;
	float scale = 5.0f;
	
    this->input = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * this->sizeX * this->sizeY, NULL, NULL);
    this->bitmap = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * this->sizeX * this->sizeY, NULL, NULL);
    this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->memSizeY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
	float *inputArray = new float[this->sizeX * this->sizeY];

    for (int x = 0; x < this->sizeX; x++)
        for (int y = 0; y < this->sizeY; y++) {
            inputArray[this->indexOf(x, y)] = 0.0f;

            if (x > sizeX / 2.0f - sizeX / scale && x < sizeX / 2.0f + sizeX / scale &&
                y > sizeY / 2.0f - sizeY / scale && y < sizeY / 2.0f + sizeY / scale)
                inputArray[this->indexOf(x, y)] = 255.0f;
        }
		
	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clEnqueueWriteBuffer(this->commands[i], this->input, CL_TRUE, 0, sizeof(float) * this->sizeX * this->sizeY, inputArray, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeY, CL_TRUE, 0, sizeof(size_t), &this->sizeY, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void HeatTransfer::performAlgorithm() {
	int error = 0;
	size_t dim = 0;
	size_t maxLocals = 0;
	size_t local[2];
	size_t global[2];
	cl_mem
		*parameter1 = &this->input,
		*parameter2 = &this->bitmap,
		*temp;

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->memSizeY);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxLocals, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}

	local[0] = this->localWorkSize(maxLocals, this->sizeX, 2);
	local[1] = this->localWorkSize(maxLocals, this->sizeY, 2);
	global[0] = this->globalWorkSize(local[0], this->sizeX);
	global[1] = this->globalWorkSize(local[1], this->sizeY);

	for (int z = 0; z < this->sizeZ; z++) {
		for (unsigned int i=0; i<deviceCount; i++) {
			error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), parameter1);
			error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), parameter2);
			if (error != CL_SUCCESS)
				this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
		}

		for (unsigned int i=0; i<deviceCount; i++) {
			error = clEnqueueNDRangeKernel(this->commands[i], this->kernel[i], 2, NULL, global, local, 0, NULL, NULL);
			if (error)
				this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
		}

		for (unsigned int i=0; i<deviceCount; i++) {
			clFinish(this->commands[i]);
		}

		temp = parameter1;
		parameter1 = parameter2;
		parameter2 = temp;
	}
}


const char *HeatTransfer::algorithm() const {
	return "															\n "\
		"__kernel void algorithm(										\n "\
		"	__global float *input,										\n "\
		"	__global float *bitmap,										\n "\
		"	__global size_t *sizeX,										\n "\
		"	__global size_t *sizeY)										\n "\
		"{																\n "\
		"	size_t x = get_global_id(0);								\n "\
		"	size_t y = get_global_id(1);								\n "\
		"																\n "\
		"	if (x < *sizeX && y < *sizeY) {								\n "\
		"		float speed = 0.25f;									\n "\
		"																\n "\
        "       int left = x ? x - 1 : 0;								\n "\
        "       int right = *sizeX - 1 < x + 1 ? *sizeX - 1 : x + 1;	\n "\
        "       int top = y ? y - 1 : 0;								\n "\
        "       int bottom = *sizeY - 1 < y + 1 ? *sizeY - 1 : y + 1;	\n "\
		"																\n "\
		"       bitmap[x * *sizeX + y] =								\n "\
		"				input[x * *sizeX + y]							\n "\
		"			+	speed * (input[x * *sizeX + top]				\n "\
		"			+	input[x * *sizeX + bottom]						\n "\
		"			+	input[left * *sizeX + y]						\n "\
		"			+	input[right * *sizeX + y]						\n "\
		"			-	input[x * *sizeX + y] * 4.0f);					\n "\
		"	}															\n "\
		"}																";
}

// Implement me!
const bool HeatTransfer::isValid() const {
	return true;
}
