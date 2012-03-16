#include "../include/MatrixMultiplicationBase.h"

const float MatrixMultiplicationBase::validationThreshold = 0.00005f;

MatrixMultiplicationBase::MatrixMultiplicationBase() {
	// this->matrixA = NULL;
	this->matrixB = NULL;
	// this->matrixC = NULL;
	// this->memSizeX = NULL;
	this->memSizeY = NULL;
	this->memSizeZ = NULL;
}

MatrixMultiplicationBase::~MatrixMultiplicationBase() {
	this->discardMembers();
}

void MatrixMultiplicationBase::discardMembers() {
	// if (this->matrixA) clReleaseMemObject(this->matrixA);
	if (this->matrixB) clReleaseMemObject(this->matrixB);
	// if (this->matrixC) clReleaseMemObject(this->matrixC);
	// if (this->memSizeX) clReleaseMemObject(this->memSizeX);
	if (this->memSizeY) clReleaseMemObject(this->memSizeY);
	if (this->memSizeZ) clReleaseMemObject(this->memSizeZ);
}

void MatrixMultiplicationBase::initializeMembers() {
	int error = 0;

	this->sizePerDevice = this->sizeZ / this->deviceCount;

	float *a = new float[this->sizeX * this->sizeZ];
	float *b = new float[this->sizeZ * this->sizeY];
	float *c = new float[this->sizeX * this->sizeY];

	for (int i = 0; i < this->sizeX; i++)
		for (int j = 0; j < this->sizeZ; j++)
			a[i * this->sizeX + j] = 
#ifdef DEBUG
				1;
#else
				this->randomFloat();
#endif
	for (int i = 0; i < this->sizeZ; i++)
		for (int j = 0; j < this->sizeY; j++)
			b[i * this->sizeZ + j] =
#ifdef DEBUG
				1;
#else 
				this->randomFloat();
#endif
	for (int i = 0; i < this->sizeX; i++)
		for (int j = 0; j < this->sizeY; j++)
			c[i * this->sizeX + j] = 0.0f;

	// create deviceCount independent buffers
	this->matrixB = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeZ * this->sizeY, NULL, NULL);
	this->memSizeX = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
	this->memSizeY = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);
	this->memSizeZ = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

	// create wrapper around matrix{A,C} for later copying of tiles
	cl_mem aBuffer = clCreateBuffer(this->context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * this->sizeX * this->sizeZ, a, &error);
	if (CL_SUCCESS != error)
		cerr << "could not create buffer for a" << endl;
	cl_mem cBuffer = clCreateBuffer(this->context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * this->sizeX * this->sizeY, c, &error);
	if (CL_SUCCESS != error)
		cerr << "could not create buffer for c" << endl;

	for (unsigned int i=0; i<deviceCount; i++) {
		workOffset[i] = i * this->sizePerDevice;
		workSize[i] = (i == deviceCount - 1) ? (this->sizeX - workOffset[i]) : sizePerDevice;
#ifdef DEBUG
		cout << "offset(" << workOffset[i] << ") size(" << workSize[i] << ")" << endl;
#endif
		this->matrixA[i] = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeX * workSize[i], NULL, NULL);
		this->matrixC[i] = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(float) * this->sizeX * workSize[i], NULL, NULL);
		// this->memSizeX[i] = clCreateBuffer(this->context,  CL_MEM_READ_WRITE, sizeof(size_t), NULL, NULL);

		error  = clEnqueueCopyBuffer(this->commands[i], aBuffer, this->matrixA[i], sizeof(float) * this->sizeX * workOffset[i], 0, sizeof(float) * this->sizeX * workSize[i], 0, NULL, NULL);
		error  = clEnqueueCopyBuffer(this->commands[i], cBuffer, this->matrixC[i], sizeof(float) * this->sizeX * workOffset[i], 0, sizeof(float) * this->sizeX * workSize[i], 0, NULL, NULL);
		// error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeX[i], CL_TRUE, 0, sizeof(size_t), &workSize[i], 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->matrixB, CL_TRUE, 0, sizeof(float) * this->sizeZ * this->sizeY, b, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeX, CL_TRUE, 0, sizeof(size_t), &this->sizeX, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeY, CL_TRUE, 0, sizeof(size_t), &this->sizeY, 0, NULL, NULL);
		error |= clEnqueueWriteBuffer(this->commands[i], this->memSizeZ, CL_TRUE, 0, sizeof(size_t), &this->sizeZ, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to write to source array for " + this->getName() + "!\n", 1);
	}

	delete a;
	delete b;
	delete c;
}

const bool MatrixMultiplicationBase::isValid() const {
	int error = 0;
	bool result = true;

	float *a = new float[this->sizeX * this->sizeZ];
	float *b = new float[this->sizeZ * this->sizeY];
	float *c = new float[this->sizeX * this->sizeY];	

	for (unsigned int d=0; d<this->deviceCount; d++) {
		float *aTile = new float[this->sizeX * workSize[d]];
		float *cTile = new float[this->sizeX * workSize[d]];

		error = clEnqueueReadBuffer(this->commands[d], this->matrixA[d], CL_TRUE, 0, sizeof(float) * this->sizeX * workSize[d], aTile, 0, NULL, NULL);
		error |= clEnqueueReadBuffer(this->commands[d], this->matrixB, CL_TRUE, 0, sizeof(float) * this->sizeZ * this->sizeY, b, 0, NULL, NULL);
		error |= clEnqueueReadBuffer(this->commands[d], this->matrixC[d], CL_TRUE, 0, sizeof(float) * this->sizeX * workSize[d], cTile, 0, NULL, NULL);
		if (error != CL_SUCCESS)
			this->exitWith("Error: Failed to read output array for " + this->getName() + "!\n", 1);

		// insert {a,c}Tile into {a,c}
		for (int i=0; i<workSize[d]; i++) { // i == row number 
			for (int k=0; k<this->sizeX; k++) {
				const int from =  i * this->sizeX + k;
				const int to = from + this->sizeX * workOffset[d];
#ifdef DEBUG
				cout << "read from " << from << " (a=" << aTile[from] << ", c="<< cTile[from] <<"), write to " << to << " (?)" << endl;
#endif
				a[to] = aTile[from];
				c[to] = cTile[from];
			}
		}

		delete aTile;
		delete cTile;
	}
#ifdef DEBUG						
	cout << "A ===========================" << endl;
	for (int i=0; i<this->sizeX*this->sizeZ; i++) {
		printf("%2.2f ", a[i]);
		if ((i+1) % this->sizeX == 0)
			printf("\n");
	}
	cout << endl;
	cout << "C ===========================" << endl;
	for (int i=0; i<this->sizeX*this->sizeY; i++) {
		printf("%2.2f ", c[i]);
		if ((i+1) % this->sizeX == 0)
			printf("\n");
	}
	cout << endl;
#endif
	result = this->validateResult(a, b, c);

	delete a;
	delete b;
	delete c;

	return result;
}

const bool MatrixMultiplicationBase::validateResult(float const* const& a, float const* const& b, float const* const& c) const {
	for (int i = 0; i < this->sizeX; i++)
		for (int j = 0; j < this->sizeY; j++)         {
			float v = 0;

			for (int k = 0; k < this->sizeZ; k++)
				v += a[i * this->sizeX + k] * b[k * this->sizeZ + j];

			if (abs(c[i * this->sizeX + j] - v) > MatrixMultiplicationBase::validationThreshold)
				return false;
		}

	return true;
}
