#ifndef DOTPRODUCT
#define DOTPRODUCT

#include "ExampleBase.h"
#include <string>

class DotProduct : public ExampleBase {

public:

	DotProduct();

	~DotProduct();

	virtual std::string getName() const;


protected:

    cl_mem matrixA;
    cl_mem matrixB;
	cl_mem memSizeX;
	cl_mem memSizeY;
	cl_mem tempResults;
	size_t *arrayMatrixA;
	size_t *arrayMatrixB;
	size_t result;

	static const float validationThreshold;

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void performAlgorithm();

	const char *algorithm() const;

	const bool isValid() const;
};

#endif