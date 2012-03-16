#ifndef CONVOLUTION
#define CONVOLUTION

#include "ExampleBase.h"
#include <string>

class Convolution : public ExampleBase {

public:

	Convolution();

	~Convolution();

	virtual std::string getName() const;


protected:

    cl_mem startImage;
	cl_mem outImage;
	cl_mem memSizeX;
	cl_mem memSizeY;

	static const float validationThreshold;

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void fillInData(float *&startImageArray, float *&outImageArray) const;

	void performAlgorithm();

	const char *algorithm() const;

	const bool isValid() const;

	const bool validateResult(float const* const& startImageArray, float const* const& outImageArray) const;
};

#endif