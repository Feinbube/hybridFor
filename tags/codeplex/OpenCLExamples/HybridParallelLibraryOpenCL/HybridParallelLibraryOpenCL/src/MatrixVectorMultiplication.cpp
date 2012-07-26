#include "../include/MatrixVectorMultiplication.h"

const float MatrixVectorMultiplication::validationThreshold = 0.00005f;

MatrixVectorMultiplication::MatrixVectorMultiplication()
	:	matrixVals(NULL),
		matrixRcIndex(NULL),
		matrixRcPointer(NULL),
		matrixNumRows(NULL),
		inputVector(NULL),
		resultVector(NULL),
		sparseInputMatrix(NULL),
		fullInputMatrix(NULL),
		fullInputVector(NULL) { }

MatrixVectorMultiplication::~MatrixVectorMultiplication() {
	this->discardMembers();
}

std::string MatrixVectorMultiplication::getName() const {
	return "MatrixVectorMultiplication";
}

void MatrixVectorMultiplication::scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) {
    double factor = 500.0;
    this->sizeX = (int) (sizeX * factor);
    this->sizeY = (int) (sizeY * factor);
}

void MatrixVectorMultiplication::discardMembers() {
    if (this->matrixVals) clReleaseMemObject(this->matrixVals);
    if (this->matrixRcIndex) clReleaseMemObject(this->matrixRcIndex);
    if (this->matrixRcPointer) clReleaseMemObject(this->matrixRcPointer);
    if (this->matrixNumRows) clReleaseMemObject(this->matrixNumRows);
    if (this->inputVector) clReleaseMemObject(this->inputVector);
    if (this->resultVector) clReleaseMemObject(this->resultVector);
	if (this->sparseInputMatrix) delete this->sparseInputMatrix;
	if (this->fullInputMatrix) delete this->fullInputMatrix;
	if (this->fullInputVector) delete this->fullInputVector;
}

void MatrixVectorMultiplication::initializeMembers() {
	int error = 0;

	this->sparseInputMatrix = new SpMat;
	this->fullInputMatrix = new float[this->sizeX * this->sizeY];
	this->fullInputVector = new float[this->sizeX];
	this->initializeInputMatrix();
	this->initializeInputVector();
	this->convertToCsr();

	this->matrixVals = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(float) * this->sparseInputMatrix->nnze, NULL, NULL);
    this->matrixRcIndex = clCreateBuffer(this->context, CL_MEM_READ_WRITE, sizeof(size_t) * this->sparseInputMatrix->nnze, NULL, NULL);
    this->matrixRcPointer = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t) * this->sizeX + 1, NULL, NULL);
    this->matrixNumRows = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
    this->inputVector = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeY, NULL, NULL);
    this->resultVector = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeX, NULL, NULL);
	
	for (cl_uint i=0; i<this->deviceCount; i++) {
		error  = clEnqueueWriteBuffer(this->commands[i], this->matrixVals, CL_TRUE, 0, sizeof(float) * this->sparseInputMatrix->nnze, this->sparseInputMatrix->val, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->matrixRcIndex, CL_TRUE, 0, sizeof(size_t) * this->sparseInputMatrix->nnze, this->sparseInputMatrix->rc_ind, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->matrixRcPointer, CL_TRUE, 0, sizeof(size_t) * this->sizeX + 1, this->sparseInputMatrix->rc_ptr, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->matrixNumRows, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->inputVector, CL_TRUE, 0, sizeof(float) * this->sizeY, this->fullInputVector, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}
}

void MatrixVectorMultiplication::initializeInputMatrix() {
	int randX, randY;

	for (int x = 0; x < this->sizeX; x++)
		for (int y = 0; y < this->sizeY; y++)
			this->fullInputMatrix[x * this->sizeX + y] = 0.0;

	for (int k = 0; k < 5 * this->sizeX; k++) {
		randX = rand() % this->sizeX;
		randY = rand() % this->sizeY;
		this->fullInputMatrix[randX * this->sizeX + randY] = this->randomFloat();
	}
}

void MatrixVectorMultiplication::initializeInputVector() {
	for (int y = 0; y < this->sizeY; y++)
		this->fullInputVector[y] = this->randomFloat();
}

void MatrixVectorMultiplication::convertToCsr() {
	size_t colCount, totalNumNonZeroes = 0;
	int x, y, k;

	this->sparseInputMatrix->nrow = this->sizeX;
	this->sparseInputMatrix->ncol = this->sizeY;
	this->sparseInputMatrix->rc_len = new size_t[this->sizeX];
	this->sparseInputMatrix->rc_ptr = new size_t[this->sizeX + 1];
	this->sparseInputMatrix->rc_ptr[0] = 0;

	for (x = 0; x < this->sizeX; x++) {
		colCount = 0;

		for (y = 0; y < this->sizeY; y++)
			if (this->fullInputMatrix[this->indexOf(x, y)] != 0.0) {
				totalNumNonZeroes++;
				colCount++;
			}

		this->sparseInputMatrix->rc_len[x] = colCount;
	}

	this->sparseInputMatrix->nnze = totalNumNonZeroes;
	this->sparseInputMatrix->rc_ind = new size_t[totalNumNonZeroes];
	this->sparseInputMatrix->val = new float[totalNumNonZeroes];

	for (x = 1; x <= this->sizeX; x++)
		this->sparseInputMatrix->rc_ptr[x] =
				this->sparseInputMatrix->rc_ptr[x - 1]
			+	this->sparseInputMatrix->rc_len[x - 1];

	k = 0;
	for (x = 0; x < this->sizeX; x++)
		for (y = 0; y < this->sizeY; y++)
			if (this->fullInputMatrix[this->indexOf(x, y)] != 0.0) {
				this->sparseInputMatrix->val[k] = this->fullInputMatrix[this->indexOf(x, y)];
				this->sparseInputMatrix->rc_ind[k] = y;
				k++;
			}
}

void MatrixVectorMultiplication::performAlgorithm() {
	int error = 0;
	size_t local = 0;
	size_t global = 0;

	for (cl_uint i=0; i<this->deviceCount; i++) {
		error  = clSetKernelArg(this->kernel[i], 0, sizeof(cl_mem), &this->matrixVals);
		error |= clSetKernelArg(this->kernel[i], 1, sizeof(cl_mem), &this->matrixRcIndex);
		error |= clSetKernelArg(this->kernel[i], 2, sizeof(cl_mem), &this->matrixRcPointer);
		error |= clSetKernelArg(this->kernel[i], 3, sizeof(cl_mem), &this->matrixNumRows);
		error |= clSetKernelArg(this->kernel[i], 4, sizeof(cl_mem), &this->inputVector);
		error |= clSetKernelArg(this->kernel[i], 5, sizeof(cl_mem), &this->resultVector);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to set kernel arguments for " + this->getName() + "!", error);
	}

	for (cl_uint i=0; i<this->deviceCount; i++) {
		error = clGetKernelWorkGroupInfo(this->kernel[i], this->device_ids[i], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to retrieve kernel work group info for " + this->getName() + "!", error);
	}

	global = this->globalWorkSize(local, this->sizeX);
	
	enqueueAndFinish(this->kernel, this->commands, &global, &local, 1);
}

const char *MatrixVectorMultiplication::algorithm() const {
	return "																					\n "\
		"__kernel void algorithm(																\n "\
		"	__global float *matrixVals,															\n "\
		"	__global size_t *matrixRcIndex,														\n "\
		"	__global size_t *matrixRcPointer,													\n "\
		"	__global size_t *matrixNumRows,														\n "\
		"	__global float *inputVector,														\n "\
		"	__global float *resultVector)														\n "\
		"{																						\n "\
		"	size_t row = get_global_id(0);														\n "\
		"																						\n "\
		"	if (row < *matrixNumRows)															\n "\
		"	{																					\n "\
		"		resultVector[row] = 0.0;														\n "\
		"		for (size_t rc = matrixRcPointer[row]; rc < matrixRcPointer[row + 1]; rc++)		\n "\
		"			resultVector[row] += matrixVals[rc] * inputVector[matrixRcIndex[rc]];		\n "\
		"	}																					\n "\
		"}																						";
}

const bool MatrixVectorMultiplication::isValid() const {
	int error;
	bool result;

    float *fullResultVector = new float[this->sizeX];

	for (cl_uint i=0; i<this->deviceCount; i++) {
		error = clEnqueueReadBuffer(this->commands[i], this->resultVector, CL_TRUE, 0, sizeof(float) * this->sizeX, fullResultVector, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);
	}

	result = this->validateResult(fullResultVector);

	delete fullResultVector;

	return result;
}

const bool MatrixVectorMultiplication::validateResult(float const* const& fullResultVector) const {
	const float *cpuFullResultVector = this->matrixVectorMultiply();
	bool result = true;

    for (int x = 1; x < this->sizeX - 1; x++)
		if(cpuFullResultVector[x] - fullResultVector[x]
			> MatrixVectorMultiplication::validationThreshold)
		{
            result = false;
			break;
		}

	delete cpuFullResultVector;
    return result;
}


const float* const MatrixVectorMultiplication::matrixVectorMultiply() const {
	float *fullResultVector = new float[this->sizeX];

	for (int x = 0; x < this->sizeX; x++) {
		fullResultVector[x] = 0.0;

		for (int y = 0; y < this->sizeY; y++)
			fullResultVector[x] +=
					this->fullInputMatrix[this->indexOf(x, y)]
				*	this->fullInputVector[y];
	}

	return fullResultVector;
}