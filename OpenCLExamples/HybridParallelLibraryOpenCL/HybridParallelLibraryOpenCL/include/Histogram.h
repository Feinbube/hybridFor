#ifndef HISTOGRAM
#define HISTOGRAM

#include "ExampleBase.h"
#include <string>

class Histogram : public ExampleBase {

public:

	Histogram();

	~Histogram();

	virtual std::string getName() const;


protected:

    cl_mem temp;
    cl_mem buffer;
    cl_mem histo;
	cl_mem memSizeX;
	cl_mem memSizeY;
	cl_mem memSizeZ;
	size_t *bufferArray;
    cl_program program2;
    cl_program program3;
	cl_kernel kernel2[MAX_GPU_COUNT];
	cl_kernel kernel3[MAX_GPU_COUNT];

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void performAlgorithm();

	const char *algorithm() const;
	const char *algorithm2() const;
	const char *algorithm3() const;

	void executeAlgorithm(size_t global, size_t local);
	void executeAlgorithm2(size_t global, size_t local);
	void executeAlgorithm3(size_t global, size_t local);

	void initializeOpenCL();

	const bool isValid() const;
};

#endif