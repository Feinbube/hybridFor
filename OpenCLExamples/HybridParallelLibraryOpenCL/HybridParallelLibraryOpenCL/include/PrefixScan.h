#ifndef PREFIXSCAN
#define PREFIXSCAN

#include "ExampleBase.h"
#include <string>

class PrefixScan : public ExampleBase {

public:

	PrefixScan();

	~PrefixScan();

	virtual std::string getName() const;


protected:

    cl_mem startData;
	cl_mem scannedData;
	cl_mem memSizeX;
	float *startDataArray;
	float *scannedDataArray;
    cl_program program2;
	cl_kernel kernel2[MAX_GPU_COUNT];

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void performAlgorithm();

	void setKernelArguments(cl_mem &incrementMemory) const;

	const char *algorithm() const;
	const char *algorithm2() const;
	
	void executeAlgorithm(size_t global, size_t local);
	void executeAlgorithm2(size_t global, size_t local);

	void initializeOpenCL();

	const bool isValid() const;
};

#endif