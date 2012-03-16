#include "../include/Ripple.h"

const float Ripple::validationThreshold = 0.00005f;

Ripple::Ripple()
	:
	memSizeX(NULL),
	memSizeY(NULL) { }

Ripple::~Ripple() {
	this->discardMembers();
}

std::string Ripple::getName() const {
	return "Ripple";
}

void Ripple::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
	float factor =  
#ifdef DEBUG
	40.0;
	// Working:
	// 4
	// 8
	//
	// Validation fails: Each time the (32+1)-th value of the first row is wrong
	// 36;
	// 64;
	// 160.0;
	// 1000;
	//
	// OUT_OF_RESOURCES error:
	// 10.0;
	// 12.0;
	// 16.0; 
	// 20.0;
	// 24.0;
	// 28.0;
	// 32.0;
#else
	8.0;
#endif
	this->sizeX = (int) (sizeX * factor);
	this->sizeY = (int) (sizeY * factor);
#ifdef DEBUG
	cout << "sizeX(" << this->sizeX << ") sizeY(" << this->sizeY << ")" << endl;
#endif
}

void Ripple::discardMembers() {
	for (unsigned int i=0; i<deviceCount; i++) {
		if (this->bitmap[i]) {
			clReleaseMemObject(this->bitmap[i]);
		}
		if (this->offset[i]) {
			clReleaseMemObject(this->offset[i]);
		}
	}
	if (this->memSizeX) clReleaseMemObject(this->memSizeX);
	if (this->memSizeY) clReleaseMemObject(this->memSizeY);
	//this->releaseMemObjects(this->bitmap);
	//this->releaseMemObjects(this->offset);
	//this->releaseMemObject(this->memSizeX);
	// this->releaseMemObject(this->memSizeY);
}

void Ripple::initializeMembers() {
	int error;
	// split work per row of work. sizePerDevice then contains the number of cells computed by each device.
	this->sizePerDevice = this->sizeX * this->sizeY / this->deviceCount;	

	this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
	this->memSizeY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

	for (unsigned int i=0; i<deviceCount; i++) {
		workOffset[i] = i * this->sizePerDevice;
		workSize[i] = (i == deviceCount - 1) ? (this->sizeX*this->sizeY - workOffset[i]) : sizePerDevice;
#ifdef DEBUG
		cout << "offset(" << workOffset[i] << ") size(" << workSize[i] << ")" << endl;
#endif
		cout << "Dev" << i << " created buffer for " << workSize[i] << " floats" << endl;

		this->bitmap[i] = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * workSize[i], NULL, NULL);
		this->offset[i] = clCreateBuffer(this->context, CL_MEM_READ_ONLY, sizeof(size_t), NULL, NULL);

		error  = clEnqueueWriteBuffer(this->commands[i], this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeY, CL_TRUE, 0, sizeof(size_t), &this->sizeY, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->offset[i], CL_TRUE, 0, sizeof(size_t), &this->workOffset[i], 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void Ripple::performAlgorithm() {
	int error = 0;
	size_t dim = 0;
	size_t maxLocals = 0;
	size_t local[2];
	size_t global[2];

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->bitmap[i]);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->memSizeY);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->offset[i]);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxLocals, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}

	// TODO fix number of local and global elements, i.e. thread count 
	local[0] = this->localWorkSize(maxLocals, this->sizeX, 2);
	global[0] = this->globalWorkSize(local[0], this->sizeX);
	local[1] = this->localWorkSize(maxLocals, this->sizeY, 2);
	global[1] = this->globalWorkSize(local[1], this->sizeY);  
	// Does not work
	// local[1] = this->localWorkSize(maxLocals, this->sizeY/deviceCount, 2);
	// global[1] = this->globalWorkSize(local[1], this->sizeY/deviceCount);  

	for (unsigned int i=0; i<deviceCount; i++) {
#ifdef DEBUG
	//	cout << "[" << i << "] " << "global0(" << global[0] << ") local0(" << local[0] << ")" << endl;	
	//	cout << "[" << i << "] " << "global1(" << global[1] << ") local1(" << local[1] << ")" << endl;	
#endif
		error = clEnqueueNDRangeKernel(this->commands[i], this->kernel[i], 2, NULL, global, local, 0, NULL, NULL);
		if (error)
			this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		clFinish(this->commands[i]);
	}
}

const char *Ripple::algorithm() const {
	return "																						\n "\
		"__kernel void algorithm(										\n"\
		"	__global float *bitmap,										\n"\
		"	__global size_t *sizeX,										\n"\
		"	__global size_t *sizeY,                                                                         \n"\
		"	__global size_t *offset)									\n"\
		"{													\n"\
		"       size_t x = get_global_id(0);									\n"\
		"       size_t y = get_global_id(1);									\n"\
		"	size_t globalY = *offset / *sizeX;								\n"\
		"													\n"\
		"       if (x < *sizeX && globalY < *sizeY) {                                                                 	\n"\
		"               int ticks = 1;                                      					\n"\
		"               float fx = x - *sizeX / 2.0f;                                                 \n"\
		"               float fy = (y+globalY) - *sizeY / 2.0f;                                                      \n"\
		"               float d = sqrt(fx * fx + fy * fy);                                                       \n"\
		"               float grey = 128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f);	\n"\
		"               bitmap[y * *sizeX + x] = grey; // *offset + (y * *sizeX + x);                                                \n"\
		"       }                                                                                                \n"\
		"}                                                                                                       \n";  
}

const bool Ripple::isValid() const {
	bool result = true;
	float *bitmapArray = new float[this->sizeX * this->sizeY];

	for (unsigned int i=0; i<this->deviceCount; i++) {
		float *tile = new float[workSize[i]];
		
		cout << "Dev" << i << " will try to read " << workSize[i] << " floats" << endl;
		int error = clEnqueueReadBuffer(this->commands[i], this->bitmap[i], CL_TRUE, 0, sizeof(float) * workSize[i], tile, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			// will receive CL_OUT_OF_RESOURCES for factor == 3000. not for factor == 1000.
			cerr << "error = " << error << endl;
			this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);
		}
#ifdef DEBUG
	//	cout << "tile: " << endl;
#endif
		for (int k=0; k<workSize[i]; k++) {
#ifdef DEBUG
	//		cout << tile[k] << " ";
			if ((k+1) % this->sizeX == 0) {
	//			cout << endl;
			}
	//		cout << "copy (" << tile[k] << ") from " << k << " to " << (workOffset[i] + k) << endl;
#endif
			bitmapArray[workOffset[i] + k] = tile[k];
		}

		// delete tile
	}

#ifdef DEBUG
	cout << "bitmap" << endl;
#endif
	for (int y = 0; y < this->sizeY; y++) {
		for (int x = 0; x < this->sizeX; x++) {
#ifdef DEBUG
			cout << bitmapArray[y * this->sizeX + x] << " ";
#endif
			int ticks = 1;
			float fx = x - this->sizeX / 2.0f;
			float fy = y - this->sizeY / 2.0f;
			float d = sqrt(fx * fx + fy * fy);
			float grey = 128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f);

			if (bitmapArray[y * sizeX + x] - grey > Ripple::validationThreshold) {
				cerr << "difference is " << (bitmapArray[y * sizeX + x] - grey)  << endl;
				result = false;
				break;
			}
		}
#ifdef DEBUG
		cout << endl;
#endif
	}

	delete bitmapArray;
	return result;
}

