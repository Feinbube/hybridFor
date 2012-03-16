#ifndef MATRIXMULTIPLICATION3
#define MATRIXMULTIPLICATION3

#include "MatrixMultiplicationBase.h"
#include <string>

class MatrixMultiplication3 : public MatrixMultiplicationBase {

public:

	MatrixMultiplication3();

	~MatrixMultiplication3();

	virtual std::string getName() const;


protected:

	virtual void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	virtual void performAlgorithm();

	const char *algorithm() const;
};

#endif