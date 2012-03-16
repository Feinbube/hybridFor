#ifndef MATRIXMULTIPLICATION1
#define MATRIXMULTIPLICATION1

#include "MatrixMultiplicationBase.h"
#include <string>

class MatrixMultiplication1 : public MatrixMultiplicationBase {

public:

	MatrixMultiplication1();

	~MatrixMultiplication1();

	virtual std::string getName() const;


protected:

	virtual void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	virtual void performAlgorithm();

	const char *algorithm() const;
};

#endif