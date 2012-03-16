#include "../include/SummingVectors.h"

SummingVectors::SummingVectors()
	:
		aArray(NULL),
		bArray(NULL) { }

SummingVectors::~SummingVectors() {
	this->discardMembers();
}

std::string SummingVectors::getName() const {
	return "SummingVectors";
}

void SummingVectors::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
	//this->sizeX = (int) (sizeX * 500.0);
	this->sizeX = (int) (sizeX * 5000000.0);
}


void SummingVectors::discardMembers() {
/*	if (this->a) clReleaseMemObject(this->a);
	if (this->b) clReleaseMemObject(this->b);
	if (this->c) clReleaseMemObject(this->c);
*/	if (this->aArray) delete this->aArray;
	if (this->bArray) delete this->bArray;
}

void SummingVectors::initializeMembers() {
	int error = 0;
	sizePerDevice = this->sizeX / this->deviceCount;

	this->aArray = new size_t[this->sizeX];
	this->bArray = new size_t[this->sizeX];
	for (int x = 0; x < this->sizeX; x++) {
		this->aArray[x] = -x;
		this->bArray[x] = x * x;
	}

	cl_mem aBuffer = clCreateBuffer(this->context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, this->sizeX * sizeof(size_t), aArray, &error);
	if (CL_SUCCESS != error)
		cerr << "could not create buffer for a" << endl;
	cl_mem bBuffer = clCreateBuffer(this->context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, this->sizeX * sizeof(size_t), bArray, &error);
	if (CL_SUCCESS != error)
		cerr << "could not create buffer for b" << endl;

	for (unsigned int i=0; i<deviceCount; i++) {
		workSize[i] = (i == deviceCount - 1) ? (this->sizeX - sizePerDevice * (deviceCount - 1)) : sizePerDevice;
		workOffset[i] = i*sizePerDevice; 
#ifdef DEBUG
		cout << "workSize(" << workSize[i] << ") workOffset(" << workOffset[i] << ")" << endl;
#endif
		this->a[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * workSize[i], NULL, NULL);
		this->b[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * workSize[i], NULL, NULL);
		this->c[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * workSize[i], NULL, NULL);
		this->len[i] = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

		error  = clEnqueueCopyBuffer(this->commands[i], aBuffer, this->a[i], sizeof(size_t) * workOffset[i], 0, sizeof(size_t) * workSize[i], 0, NULL, NULL);
		error |= clEnqueueCopyBuffer(this->commands[i], bBuffer, this->b[i], sizeof(size_t) * workOffset[i], 0, sizeof(size_t) * workSize[i], 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->len[i], CL_TRUE, 0, sizeof(size_t), &workSize[i], 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void SummingVectors::performAlgorithm() {
	int error = 0;
	size_t local = 0;
	size_t global = 0;

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->a[i]);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->b[i]);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->c[i]);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->len[i]);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
#ifdef DEBUG
		cout << "[" << i << "] set local to " << local << endl;
#endif
	}
	// TODO Muessen local und global nun pro Device neu berechnet werden?
	global = this->globalWorkSize(local, this->sizeX);
#ifdef DEBUG
	cout << "set global to " << global << endl;
#endif
	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueNDRangeKernel(this->commands[i], this->kernel[i], 1, NULL, &global, &local, 0, NULL, NULL);
		if (error)
			this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		clFinish(this->commands[i]);
	}
}

const char *SummingVectors::algorithm() const {
	return "								\n "\
		"__kernel void algorithm(			\n "\
		"	__global size_t *a,				\n "\
		"	__global size_t *b,				\n "\
		"	__global size_t *c,				\n "\
		"	__global size_t *sizeX)			\n "\
		"{									\n "\
		"	size_t x = get_global_id(0);	\n "\
		"									\n "\
		"	if (x < *sizeX) {				\n "\
		"		c[x] = a[x] + b[x];			\n "\
		"	}								\n "\
		"}									";
}

const bool SummingVectors::isValid() const {
	int error;
	bool result = true;

	for (unsigned int i=0; i<deviceCount; i++) {
		size_t *cArray = new size_t[workSize[i]];
		printf("copying from #0 .. #%d\n", workSize[i]); 

		error = clEnqueueReadBuffer(this->commands[0], this->c[i], CL_TRUE, 0, sizeof(size_t) * workSize[i], cArray, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);
		
		for (int x = 0; x < workSize[i]; x++) {
#ifdef DEBUG
			printf("[%d/%d] = %d\n", i, x, cArray[x]);
			printf("[%d/%d] workOffset[%d] == %d\n", i, x, i, workOffset[i]);
			printf("[%d/%d] %d + %d != %d\n", i, x, aArray[workOffset[i] + x], bArray[workOffset[i] + x], cArray[x]);
#endif
			if (cArray[x] != this->aArray[workOffset[i] + x] + this->bArray[workOffset[i] + x]) {
				result = false;
				break;
			}
		}

		delete[] cArray;
	}

	return result;
}
