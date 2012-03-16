#ifndef MATRIXMULTIPLICATION4
#define MATRIXMULTIPLICATION4

#include "MatrixMultiplicationBase.h"
#include <string>

class MatrixMultiplication4 : public MatrixMultiplicationBase {

public:

	MatrixMultiplication4();

	~MatrixMultiplication4();

	virtual std::string getName() const;


protected:

	virtual void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	virtual void performAlgorithm();

	const char *algorithm() const;
};

#endif