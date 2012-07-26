#include "../include/Dummy.h"

Dummy::Dummy()/*:
		in(NULL),
		out(NULL),
		inArray(NULL),
		outArray(NULL)*/ { }

Dummy::~Dummy() {
	this->discardMembers();
}

std::string Dummy::getName() const {
	return "Dummy";
}

void Dummy::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
	this->sizeX = 10;
}

void Dummy::discardMembers() {
/*	for (unsigned int i=0; i<deviceCount; i++) {
		if (this->in[i]) clReleaseMemObject(this->in[i]);
		if (this->out[i]) clReleaseMemObject(this->out[i]);
	}
*/
//	if (this->inArray) delete this->inArray;
//	if (this->outArray) delete this->outArray;

	// for (unsigned int i=0; i<deviceCount; i++)
	// 	if (this->offset[i])
	//	clReleaseMemObject(this->offset[i]);
}

void Dummy::initializeMembers() {
	int error = 0;

	sizePerDevice = this->sizeX / this->deviceCount;

	this->inArray = new float[this->sizeX];
	this->outArray = new float[this->sizeX];

	for (int x = 0; x < this->sizeX; x++) {
		this->inArray[x] = (float) x;
		this->outArray[x] = (float) 0;
	}

	cl_mem inBuffer = clCreateBuffer(this->context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, this->sizeX * sizeof(float), inArray, &error);
	if (CL_SUCCESS != error)
		cerr << "could not create buffer for in" << endl;
	cl_mem outBuffer = clCreateBuffer(this->context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, this->sizeX * sizeof(float), outArray, &error);
	if (CL_SUCCESS != error)
		cerr << "could not create buffer for out" << endl;

	for (unsigned int i=0; i<deviceCount; i++) {
		workSize[i] = (i == deviceCount - 1) ? (this->sizeX - sizePerDevice * (deviceCount - 1)) : sizePerDevice;
		workOffset[i] = (i == 0) ? (0) : (sizePerDevice + workOffset[i - 1]); 
#ifdef DEBUG
		cout << "workSize(" << workSize[i] << ") workOffset(" << workOffset[i] << ")" << endl;
#endif
		this->in[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * workSize[i], NULL, NULL);
		this->out[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * workSize[i], NULL, NULL);
		this->len[i] = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

		error  = clEnqueueCopyBuffer(this->commands[i], inBuffer, this->in[i], sizeof(float) * workOffset[i], 0, sizeof(float) * workSize[i], 0, NULL, NULL);
		error |= clEnqueueCopyBuffer(this->commands[i], outBuffer, this->out[i], sizeof(float) * workOffset[i], 0, sizeof(float) * workSize[i] , 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->len[i], CL_TRUE, 0, sizeof(size_t), &workSize[i], 0, NULL, NULL);
	/*	
		   error  = clEnqueueWriteBuffer(this->commands[i], this->in[i], CL_TRUE, sizeof(float) * workOffset[i], sizeof(float) * workSize[i], inArray, 0, NULL, NULL);
		   error |= clEnqueueWriteBuffer(this->commands[i], this->out[i], CL_TRUE, sizeof(float) * workOffset[i], sizeof(float) * workSize[i] , outArray, 0, NULL, NULL);
		   error |= clEnqueueWriteBuffer(this->commands[i], this->len[i], CL_TRUE, 0, sizeof(size_t), &workSize[i], 0, NULL, NULL);
	*/	 
		if (CL_SUCCESS != error)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void Dummy::performAlgorithm() {
	int error = 0;
	size_t local = 0;
	size_t global = 0;

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->in[i]);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->out[i]);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->len[i]);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
		global = this->globalWorkSize(local, this->sizeX);
	}

	//	cout << "global: " << global << endl;
	// global = global / deviceCount;
	//global = 4;
	//local = 1;
	//	cout << "reduced global to " << global << endl;
	// TODO is it correct to reduce the global work size??

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueNDRangeKernel(this->commands[i], this->kernel[i], 1, 0, &global, &local, 0, NULL, NULL);
		if (error)
			this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		clFinish(this->commands[i]);
	}
}

const char *Dummy::algorithm() const {
	return "												\n "\
		"__kernel void algorithm(							\n "\
		"	__global float *in,								\n "\
		"	__global float *out,								\n "\
		"	__global size_t *len)								\n "\
		"{													\n "\
		"	size_t x = get_global_id(0);					\n "\
		"													\n "\
		"	if (x < *len) {								\n "\
		"		out[x] = 2 * in[x];					\n "\
		"	}												\n "\
		"}													";
}

const bool Dummy::isValid() const {
	int error;
	bool result = true;
	float *resultArray = new float[this->sizeX];
#ifdef DEBUG
	for (unsigned int i=0; i<deviceCount; i++) {  
		cout << "offset(" << (sizeof(float)*workOffset[i]) << ") size(" << (sizeof(float) * workSize[i]) << ")" << endl;
	}
#endif
	for (unsigned int i=0; i<deviceCount; i++) {  
		// read without offset since data starts directly at the beginning of out[i]
		error = clEnqueueReadBuffer(this->commands[i], this->out[i], CL_TRUE, 0, sizeof(float) * workSize[i], resultArray, 0, NULL, NULL);
		for (int k=0; k<workSize[i]; k++) {
			if ((int)resultArray[k] % 2 != 0) {
				result = false; // all digits in out have to be even
			}
#ifdef DEBUG
			cout << "resultArray[" << k << "] = " << resultArray[k] << endl;
#endif
		}
		if (error != CL_SUCCESS) {
			cerr << "error " << error << endl;
			this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);
		}
	}

	delete[] resultArray;
	return result;
}
