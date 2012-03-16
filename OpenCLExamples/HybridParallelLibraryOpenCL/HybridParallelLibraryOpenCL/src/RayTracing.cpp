/*#include "../include/RayTracing.h"

RayTracing::RayTracing()
	:	bitmap(NULL),
		memSizeX(NULL),
		memSizeY(NULL),
		memSizeZ(NULL),
		spheresR(NULL),
		spheresRadius(NULL),
		spheresX(NULL),
		spheresY(NULL),
		spheresZ(NULL),
		spheres(NULL) { }

RayTracing::~RayTracing() {
	this->discardMembers();
}

std::string RayTracing::getName() const {
	return "RayTracing";
}

void RayTracing::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
	float factor = 500.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
    this->sizeZ = (int) (sizeZ * factor);
}

void RayTracing::discardMembers() {
    if (this->bitmap) clReleaseMemObject(this->bitmap);
    if (this->memSizeX) clReleaseMemObject(this->memSizeX);
    if (this->memSizeY) clReleaseMemObject(this->memSizeY);
    if (this->memSizeZ) clReleaseMemObject(this->memSizeZ);
    if (this->spheresR) clReleaseMemObject(this->spheresR);
    if (this->spheresRadius) clReleaseMemObject(this->spheresRadius);
    if (this->spheresX) clReleaseMemObject(this->spheresX);
    if (this->spheresY) clReleaseMemObject(this->spheresY);
    if (this->spheresZ) clReleaseMemObject(this->spheresZ);
	if (this->spheres) delete spheres;
}

void RayTracing::initializeMembers() {
	int error;

    this->bitmap = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeX * this->sizeY, NULL, NULL);
    this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->memSizeY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->memSizeZ = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

    this->spheresR = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeZ, NULL, NULL);
    this->spheresRadius = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeZ, NULL, NULL);
    this->spheresX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeZ, NULL, NULL);
    this->spheresY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeZ, NULL, NULL);
    this->spheresZ = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeZ, NULL, NULL);

	float *spheresRarray = new float[this->sizeZ];
	float *spheresRadiusArray = new float[this->sizeZ];
	float *spheresXarray = new float[this->sizeZ];
	float *spheresYarray = new float[this->sizeZ];
	float *spheresZarray = new float[this->sizeZ];
	
	this->spheres = new Sphere[this->sizeZ];
	for (int z = 0; z < this->sizeZ; z++) {
		this->spheres[z].r = spheresRarray[z] = this->randomFloat();
        this->spheres[z].x = spheresXarray[z] = this->randomFloat() * 100.0f - 50.0f;
        this->spheres[z].y = spheresYarray[z] = this->randomFloat() * 100.0f - 50.0f;
        this->spheres[z].z = spheresZarray[z] = this->randomFloat() * 100.0f - 50.0f;
        this->spheres[z].radius = spheresRadiusArray[z] = this->randomFloat() * 10.0f + 2.0f;
	}

	error  = clEnqueueWriteBuffer(this->commands, this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(this->commands, this->memSizeY, CL_TRUE, 0, sizeof(size_t), &this->sizeY, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(this->commands, this->memSizeZ, CL_TRUE, 0, sizeof(size_t), &this->sizeZ, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(this->commands, this->spheresR, CL_TRUE, 0, sizeof(float) * this->sizeZ, spheresRarray, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(this->commands, this->spheresRadius, CL_TRUE, 0, sizeof(float) * this->sizeZ, spheresRadiusArray, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(this->commands, this->spheresX, CL_TRUE, 0, sizeof(float) * this->sizeZ, spheresXarray, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(this->commands, this->spheresY, CL_TRUE, 0, sizeof(float) * this->sizeZ, spheresYarray, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(this->commands, this->spheresZ, CL_TRUE, 0, sizeof(float) * this->sizeZ, spheresZarray, 0, NULL, NULL);
	if (error != CL_SUCCESS)
		this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);

	delete spheresRarray;
	delete spheresRadiusArray;
	delete spheresXarray;
	delete spheresYarray;
	delete spheresZarray;
}

void RayTracing::performAlgorithm() {
	int error = 0;
	size_t dim = 0;
	size_t maxLocals = 0;
	size_t local[2];
	size_t global[2];

    error  = clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &this->bitmap);
    error |= clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &this->memSizeX);
    error |= clSetKernelArg(this->kernel, 2, sizeof(cl_mem), &this->memSizeY);
    error |= clSetKernelArg(this->kernel, 3, sizeof(cl_mem), &this->memSizeZ);
    error |= clSetKernelArg(this->kernel, 4, sizeof(cl_mem), &this->spheresR);
    error |= clSetKernelArg(this->kernel, 5, sizeof(cl_mem), &this->spheresRadius);
    error |= clSetKernelArg(this->kernel, 6, sizeof(cl_mem), &this->spheresX);
    error |= clSetKernelArg(this->kernel, 7, sizeof(cl_mem), &this->spheresY);
    error |= clSetKernelArg(this->kernel, 8, sizeof(cl_mem), &this->spheresZ);
    if (error != CL_SUCCESS)
        this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);

    error = clGetKernelWorkGroupInfo(this->kernel, this->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxLocals, NULL);
    if (error != CL_SUCCESS)
        this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);

	local[0] = this->localWorkSize(maxLocals, this->sizeX, 2);
	local[1] = this->localWorkSize(maxLocals, this->sizeY, 2);
	global[0] = this->globalWorkSize(local[0], this->sizeX);
	global[1] = this->globalWorkSize(local[1], this->sizeY);

    error = clEnqueueNDRangeKernel(this->commands, this->kernel, 2, NULL, global, local, 0, NULL, NULL);
    if (error)
        this->exitWith("Error: Failed to execute kernel for " + this->getName() + "!", error);

    clFinish(this->commands);
}

const char *RayTracing::algorithm() const {
	return "													\n "\
		"float hit(												\n "\
		"	float ox,											\n "\
		"	float oy,											\n "\
		"	float *n,											\n "\
		"	float x,											\n "\
		"	float y,											\n "\
		"	float z,											\n "\
		"	float radius)										\n "\
		"{														\n "\
        "	float												\n "\
		"		dx = ox - x,									\n "\
		"		dx2 = dx * dx,									\n "\
		"		dy = oy - y,									\n "\
		"		dy2 = dy * dy,									\n "\
		"		radius2 = radius * radius;						\n "\
		"														\n "\
        "	if (dx2 + dy2 < radius2) {							\n "\
        "        float dz = sqrt(radius2 - dx2 - dy2);			\n "\
        "        *n = dz / sqrt(radius2);						\n "\
        "        return dz + z;									\n "\
        "	}													\n "\
		"														\n "\
        "	return FLT_MIN;										\n "\
		"}														\n "\
		"														\n "\
		"__kernel void algorithm(								\n "\
		"	__global size_t *bitmap,							\n "\
		"	__global size_t *sizeX,								\n "\
		"	__global size_t *sizeY,								\n "\
		"	__global size_t *sizeZ,								\n "\
		"	__global float *spheresR,							\n "\
		"	__global float *spheresRadius,						\n "\
		"	__global float *spheresX,							\n "\
		"	__global float *spheresY,							\n "\
		"	__global float *spheresZ)							\n "\
		"{														\n "\
		"	size_t x = get_global_id(0);						\n "\
		"	size_t y = get_global_id(1);						\n "\
		"														\n "\
		"	if (x < *sizeX && y < *sizeY) {						\n "\
		"		float ox, oy, r, maxz, n, t, fscale;			\n "\
		"														\n "\
		"		ox = x - *sizeX / 2.0f;							\n "\
		"		oy = y - *sizeY / 2.0f;							\n "\
		"		r = 0.0f;										\n "\
		"		maxz = FLT_MIN;									\n "\
		"														\n "\
		"		for (int z = 0; z < *sizeZ; z++) {				\n "\
		"			n = 0.0f;									\n "\
		"			t = hit(									\n "\
		"				ox, oy, &n,								\n "\
		"				spheresX[z],							\n "\
		"				spheresY[z],							\n "\
		"				spheresZ[z],							\n "\
		"				spheresRadius[z]);						\n "\
		"														\n "\
		"			if (t > maxz) {								\n "\
		"				fscale = n;								\n "\
		"				r = spheresR[z] * fscale;				\n "\
		"			}											\n "\
		"		}												\n "\
		"														\n "\
		"		bitmap[x * *sizeX + y] = (int) (r * 256.0f);	\n "\
		"	}													\n "\
		"}														";
}

const bool RayTracing::isValid() const {
	bool result = true;
	float ox, oy, r, maxz, n, t, fscale;

	size_t *bitmapArray = new size_t[this->sizeX * this->sizeY];
    int error = clEnqueueReadBuffer(this->commands, this->bitmap, CL_TRUE, 0, sizeof(size_t) * this->sizeX * this->sizeY, bitmapArray, 0, NULL, NULL);
    if (error != CL_SUCCESS)
		this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);

    for (int x = 0; x < this->sizeX; x++) {
        for (int y = 0; y < this->sizeY; y++) {
			ox = x - this->sizeX / 2.0f,
			oy = y - this->sizeY / 2.0f,
			r = 0.0f,
			maxz = FLT_MIN;

            for (int z = 0; z < this->sizeZ; z++) {
				n = 0.0f,
				t = this->spheres[z].hit(ox, oy, n);

                if (t > maxz) {
                    fscale = n;
                    r = this->spheres[z].r * fscale;
                }
            }

			if (bitmapArray[this->indexOf(x, y)] != (int) (r * 256.0f)) {
                result = false;
				break;
			}
        }

		if (!result) break;
	}

	delete bitmapArray;
    return result;
}*/