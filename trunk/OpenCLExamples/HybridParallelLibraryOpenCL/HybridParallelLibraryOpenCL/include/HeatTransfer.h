#ifndef HEATTRANSFER
#define HEATTRANSFER

#include "ExampleBase.h"
#include <string>

class HeatTransfer : public ExampleBase {

public:

	HeatTransfer();

	~HeatTransfer();

	virtual std::string getName() const;


protected:
	
    cl_mem input;
    cl_mem bitmap;
	cl_mem memSizeX;
	cl_mem memSizeY;
	float *inputArray;

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void performAlgorithm();

	const char *algorithm() const;

	const bool isValid() const;
};

#endif