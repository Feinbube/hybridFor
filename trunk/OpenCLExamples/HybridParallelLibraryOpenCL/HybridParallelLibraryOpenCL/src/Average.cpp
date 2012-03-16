#include "../include/Average.h"

const float Average::validationThreshold = 0.0005f;

Average::Average()
	:
		aArray(NULL),
		bArray(NULL) { }

		Average::~Average() {
			this->discardMembers();
		}

std::string Average::getName() const {
	return "Average";
}

void Average::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
//	this->sizeX = (int) (sizeX * 100.0);
	this->sizeX = (int) (sizeX * 10000000.0);
}

void Average::discardMembers() {
	/*if (this->a) clReleaseMemObject(this->a);
	if (this->b) clReleaseMemObject(this->b);
	if (this->c) clReleaseMemObject(this->c);
*/	if (this->aArray) delete this->aArray;
	if (this->bArray) delete this->bArray;

	// for (unsigned int i=0; i<deviceCount; i++)
	// 	if (this->offset[i])
	//	clReleaseMemObject(this->offset[i]);
}

void Average::initializeMembers() {
	int error = 0;
	this->sizePerDevice = this->sizeX / this->deviceCount;

	this->aArray = new float[this->sizeX];
	this->bArray = new float[this->sizeX];
	for (int x = 0; x < this->sizeX; x++) {
		this->aArray[x] = this->randomFloat() * 1000;
		this->bArray[x] = this->randomFloat() * 1000;
	}

	cl_mem aBuffer = clCreateBuffer(this->context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, this->sizeX * sizeof(float), aArray, &error);
	if (CL_SUCCESS != error)
		cerr << "could not create buffer for aArray" << endl;
	cl_mem bBuffer = clCreateBuffer(this->context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, this->sizeX * sizeof(float), bArray, &error);
	if (CL_SUCCESS != error)
		cerr << "could not create buffer for bArray" << endl;

	for (unsigned int i=0; i<deviceCount; i++) {
		workSize[i] = (i == deviceCount - 1) ? (this->sizeX - sizePerDevice * (deviceCount - 1)) : sizePerDevice;
		workOffset[i] = (i == 0) ? (0) : (sizePerDevice + workOffset[i - 1]); 
#ifdef DEBUG
		cout << "workSize(" << workSize[i] << ") workOffset(" << workOffset[i] << ")" << endl;
#endif
		this->a[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float)*workSize[i], NULL, NULL);
		this->b[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float)*workSize[i], NULL, NULL);
		this->c[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float)*workSize[i], NULL, NULL);
		this->len[i] = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

		error  = clEnqueueCopyBuffer(this->commands[i], aBuffer, this->a[i], sizeof(float)*workOffset[i], 0, sizeof(float)*workSize[i], 0, NULL, NULL);
		error |= clEnqueueCopyBuffer(this->commands[i], bBuffer, this->b[i], sizeof(float)*workOffset[i], 0, sizeof(float)*workSize[i], 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->len[i], CL_TRUE, 0, sizeof(size_t), &workSize[i], 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void Average::performAlgorithm() {
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
		global = this->globalWorkSize(local, this->sizeX);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueNDRangeKernel(this->commands[i], this->kernel[i], 1, NULL, &global, &local, 0, NULL, NULL);
		if (error)
			this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		clFinish(this->commands[i]);
	}
}

const char *Average::algorithm() const {
	return "												\n "\
		"__kernel void algorithm(							\n "\
		"	__global float *a,								\n "\
		"	__global float *b,								\n "\
		"	__global float *c,								\n "\
		"	__global size_t *sizeX)								\n "\
		"{													\n "\
		"	size_t x = get_global_id(0);					\n "\
		"													\n "\
		"	if (x < *sizeX) {								\n "\
		"		size_t x1 = (x + 1) % *sizeX;				\n "\
		"		size_t x2 = (x + 2) % *sizeX;				\n "\
		"		float as = (a[x] + a[x1] + a[x2]) / 3.0f;	\n "\
		"		float bs = (b[x] + b[x1] + b[x2]) / 3.0f;	\n "\
		"		c[x] = (as + bs) / 2.0f;					\n "\
		"	}												\n "\
		"}													";
}

const bool Average::isValid() const {
	int error;
	float *cArray = new float[this->sizeX];
	bool result = true;
#ifdef DEBUG
	for (unsigned int i=0; i<deviceCount; i++) {  
		cout << "offset(" << (sizeof(float)*workOffset[i]) << ") size(" << (sizeof(float) * workSize[i]) << ")" << endl;
	}
#endif
	for (unsigned int i=0; i<deviceCount; i++) {   
		error = clEnqueueReadBuffer(this->commands[i], this->c[i], CL_TRUE, 0, sizeof(float) * workSize[i], cArray, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			cerr << "error " << error << endl;
			this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);
		}

		// TODO Das muss an den Rändern von cArray scheitern, d.h. ab x = workSize - 2
		// TODO Die Modulo Operation liefert dann einen Index zum Wert am Anfang des Teilintervalls statt zum Anfang des Gesamtintervalls.
		for (int x = 0; x < workSize[i]; x++) {
			int x1 = (x + 1) % workSize[i];
			int x2 = (x + 2) % workSize[i];

			float as = (this->aArray[x] + this->aArray[x1] + this->aArray[x2]) / 3.0f;
			float bs = (this->bArray[x] + this->bArray[x1] + this->bArray[x2]) / 3.0f;
			float cx = (as + bs) / 2.0f;

			if (cArray[x] - cx > Average::validationThreshold) {
				cerr << "[" << i << "] difference is greater than threshold: " << (cArray[x] - cx) << " > " << Average::validationThreshold << endl; 
				result = false;
				break;
			}
		}
	}

	delete[] cArray;
	return result;
}
