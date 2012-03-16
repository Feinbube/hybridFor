#include "../include/JuliaSet.h"

JuliaSet::JuliaSet()
	:	bitmap(NULL),
		memSizeX(NULL),
		memSizeY(NULL) { }

JuliaSet::~JuliaSet() {
	this->discardMembers();
}

std::string JuliaSet::getName() const {
	return "JuliaSet";
}

void JuliaSet::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
	float factor = 300.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
}

void JuliaSet::discardMembers() {
    if (this->bitmap) clReleaseMemObject(this->bitmap);
    if (this->memSizeX) clReleaseMemObject(this->memSizeX);
    if (this->memSizeY) clReleaseMemObject(this->memSizeY);
}

void JuliaSet::initializeMembers() {
	int error;

    this->bitmap = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeX * this->sizeY, NULL, NULL);
    this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->memSizeY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clEnqueueWriteBuffer(this->commands[i], this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeY, CL_TRUE, 0, sizeof(size_t), &this->sizeY, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void JuliaSet::performAlgorithm() {
	int error = 0;
	size_t dim = 0;
	size_t maxLocals = 0;
	size_t local[2];
	size_t global[2];

	for (unsigned int i=0; i<deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->bitmap);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->memSizeY);
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

	for (unsigned int i=0; i<deviceCount; i++) {
		error = clEnqueueNDRangeKernel(this->commands[i], this->kernel[i], 2, NULL, global, local, 0, NULL, NULL);
		if (error)
			this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);
	}

	for (unsigned int i=0; i<deviceCount; i++) {
		clFinish(this->commands[i]);
	}
}

const char *JuliaSet::algorithm() const {
	return "																		\n "\
		"__kernel void algorithm(													\n "\
		"	__global size_t *bitmap,												\n "\
		"	__global size_t *sizeX,													\n "\
		"	__global size_t *sizeY)													\n "\
		"{																			\n "\
		"	size_t x = get_global_id(0);											\n "\
		"	size_t y = get_global_id(1);											\n "\
		"																			\n "\
		"	if (x < *sizeX && y < *sizeY) {											\n "\
		"		const float scale = 1.5f;											\n "\
		"																			\n "\
		"		float jx = scale * (float) (*sizeX / 2.0f - x) / (*sizeX / 2.0f);	\n "\
		"		float jy = scale * (float) (*sizeY / 2.0f - y) / (*sizeY / 2.0f);	\n "\
		"																			\n "\
		"		float cr = -0.8f, ci = 0.156f;										\n "\
		"		float ar = jx, ai = jy;												\n "\
		"																			\n "\
		"		size_t v = 1;														\n "\
		"		for (int i = 0; i < 200; i++) {										\n "\
		"			float nr = ar * ar - ai * ai, ni = ai * ar + ar * ai;			\n "\
		"			ar = nr + cr; ai = ni + ci;										\n "\
		"																			\n "\
		"			if (ar * ar + ai * ai > 1000) {									\n "\
		"				v = 0;														\n "\
		"				break;														\n "\
		"			}																\n "\
		"		}																	\n "\
		"																			\n "\
		"		bitmap[x * *sizeX + y] = v;											\n "\
		"	}																		\n "\
		"}																			";
}

// Fails for most x/y/z sizes, yet not for some specific -> unprecise floats?
const bool JuliaSet::isValid() const {
	size_t *bitmapArray = new size_t[this->sizeX * this->sizeY];
	bool result = true;

	// TODO iterate across GPU results
    int error = clEnqueueReadBuffer(this->commands[0], this->bitmap, CL_TRUE, 0, sizeof(size_t) * this->sizeX * this->sizeY, bitmapArray, 0, NULL, NULL);
    if (error != CL_SUCCESS)
		this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);

	for (int x = 0; x < this->sizeX; x++) {
		for (int y = 0; y < this->sizeY; y++)
			if (bitmapArray[this->indexOf(x, y)] != this->julia(x, y)) {
				result = false;
				break;
			}

		if (!result)
			break;
	}

	delete bitmapArray;
	return result;
}

const size_t JuliaSet::julia(const int x, const int y) const {
	const float scale = 1.5f;
	size_t result = 1;
	
    float jx = scale * (float) (this->sizeX / 2.0f - x) / (this->sizeX / 2.0f);
    float jy = scale * (float) (this->sizeY / 2.0f - y) / (this->sizeY / 2.0f);
	
    Complex c = Complex(-0.8f, 0.156f);
    Complex a = Complex(jx, jy);

	for (int i = 0; i < 200; i++) {
		a = a * a + c;

		if (a.magnitude2() > 1000) {
			result = 0;
			break;
		}
	}

	return result;
}
