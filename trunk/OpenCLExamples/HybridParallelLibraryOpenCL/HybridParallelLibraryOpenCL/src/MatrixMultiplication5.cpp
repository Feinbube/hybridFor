#include "../include/MatrixMultiplication5.h"

MatrixMultiplication5::MatrixMultiplication5() {
	this->mf = NULL;
	this->ml = NULL;
	this->nf = NULL;
	this->nl = NULL;
	this->pf = NULL;
	this->pl = NULL;
}

MatrixMultiplication5::~MatrixMultiplication5() {
	this->discardMembers();
}

void MatrixMultiplication5::discardMembers() {
	MatrixMultiplicationBase::discardMembers();
}

void MatrixMultiplication5::initializeMembers() {
	MatrixMultiplicationBase::initializeMembers();
	this->mf = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	this->ml = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	this->nf = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	this->nl = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	this->pf = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	this->pl = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
}

std::string MatrixMultiplication5::getName() const {
	return "MatrixMultiplication5";
}

void MatrixMultiplication5::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
    double factor = 50.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
    this->sizeZ = (int) (sizeZ * factor);
}

void MatrixMultiplication5::performAlgorithm() {
    this->matmultrec(0, this->sizeX, 0, this->sizeY, 0, this->sizeZ);
}

void MatrixMultiplication5::setRanges(int mf, int ml, int nf, int nl, int pf, int pl) const
{
	int error;

	for (cl_uint i=0; i<this->deviceCount; i++) {
		error  = clEnqueueWriteBuffer(this->commands[i], this->mf, CL_TRUE, 0, sizeof(int), &mf, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->ml, CL_TRUE, 0, sizeof(int), &ml, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->nf, CL_TRUE, 0, sizeof(int), &nf, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->nl, CL_TRUE, 0, sizeof(int), &nl, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->pf, CL_TRUE, 0, sizeof(int), &pf, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->pl, CL_TRUE, 0, sizeof(int), &pl, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void MatrixMultiplication5::matmultrec(int mf, int ml, int nf, int nl, int pf, int pl) {
    if ((ml - mf) * (nl - nf) * (pl - pf) < 8 * 32768)
        this->matmultleaf(mf, ml, nf, nl, pf, pl);
    else {
        this->matmultrec(mf, mf + (ml - mf) / 2, nf, nf + (nl - nf) / 2, pf, pf + (pl - pf) / 2);
        this->matmultrec(mf, mf + (ml - mf) / 2, nf + (nl - nf) / 2, nl, pf, pf + (pl - pf) / 2);
        this->matmultrec(mf, mf + (ml - mf) / 2, nf, nf + (nl - nf) / 2, pf + (pl - pf) / 2, pl);
        this->matmultrec(mf, mf + (ml - mf) / 2, nf + (nl - nf) / 2, nl, pf + (pl - pf) / 2, pl);
        this->matmultrec(mf + (ml - mf) / 2, ml, nf, nf + (nl - nf) / 2, pf, pf + (pl - pf) / 2);
        this->matmultrec(mf + (ml - mf) / 2, ml, nf + (nl - nf) / 2, nl, pf, pf + (pl - pf) / 2);
        this->matmultrec(mf + (ml - mf) / 2, ml, nf, nf + (nl - nf) / 2, pf + (pl - pf) / 2, pl);
        this->matmultrec(mf + (ml - mf) / 2, ml, nf + (nl - nf) / 2, nl, pf + (pl - pf) / 2, pl);
    }
}

void MatrixMultiplication5::matmultleaf(int mf, int ml, int nf, int nl, int pf, int pl) {
	int error = 0;
	size_t maxLocals;

	this->setRanges(mf, ml, nf, nl, pf, pl);

	for (cl_uint i=0; i<this->deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->matrixA);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->matrixB);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->matrixC);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->memSizeX);
		error |= clSetKernelArg(this->kernel[i], 4, sizeof(cl_mem), &this->memSizeY);
		error |= clSetKernelArg(this->kernel[i], 5, sizeof(cl_mem), &this->memSizeZ);
		error |= clSetKernelArg(this->kernel[i], 6, sizeof(cl_mem), &this->mf);
		error |= clSetKernelArg(this->kernel[i], 7, sizeof(cl_mem), &this->ml);
		error |= clSetKernelArg(this->kernel[i], 8, sizeof(cl_mem), &this->nf);
		error |= clSetKernelArg(this->kernel[i], 9, sizeof(cl_mem), &this->nl);
		error |= clSetKernelArg(this->kernel[i], 10, sizeof(cl_mem), &this->pf);
		error |= clSetKernelArg(this->kernel[i], 11, sizeof(cl_mem), &this->pl);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (cl_uint i=0; i<this->deviceCount; i++) {
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxLocals, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}

	this->local[0] = this->localWorkSize(maxLocals, this->sizeX, 2);
	this->local[1] = this->localWorkSize(maxLocals, this->sizeY, 2);
	this->global[0] = this->globalWorkSize(this->local[0], this->sizeX);
	this->global[1] = this->globalWorkSize(this->local[1], this->sizeY);

	enqueueAndFinish(this->kernel, this->commands, this->global, this->local, 2);
}

const char *MatrixMultiplication5::algorithm() const {
	return "																				\n "\
		"__kernel void algorithm(															\n "\
		"	__global float *inputMatrixA,													\n "\
		"	__global float *inputMatrixB,													\n "\
		"	__global float *resultMatrixC,													\n "\
		"	__global size_t *m,																\n "\
		"	__global size_t *n,																\n "\
		"	__global size_t *p,																\n "\
		"	__global size_t *mf,															\n "\
		"	__global size_t *ml,															\n "\
		"	__global size_t *nf,															\n "\
		"	__global size_t *nl,															\n "\
		"	__global size_t *pf,															\n "\
		"	__global size_t *pl)															\n "\
		"{																					\n "\
		"	size_t idx = get_global_id(0);													\n "\
		"	size_t idy = get_global_id(1);													\n "\
		"																					\n "\
		"	if(idx >= *mf && idx < *ml && idy >= *nf && idy < *nl)							\n "\
		"	{																				\n "\
		"		resultMatrixC[idx * *m + idy] = 0;											\n "\
		"		for(int k = *pf; k < *pl; k++)												\n "\
		"			resultMatrixC[idx * *m + idy] +=										\n "\
		"				inputMatrixA[idx * *m + k] * inputMatrixB[k * *p + idy];			\n "\
		"	}																				\n "\
		"}																					";
}