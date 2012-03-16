#ifndef MATRIXMULTIPLICATION2
#define MATRIXMULTIPLICATION2

#include "MatrixMultiplicationBase.h"
#include <string>

class MatrixMultiplication2 : public MatrixMultiplicationBase {

public:

	MatrixMultiplication2();

	~MatrixMultiplication2();

	virtual std::string getName() const;


protected:

	virtual void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	virtual void performAlgorithm();

	const char *algorithm() const;
};

#endif