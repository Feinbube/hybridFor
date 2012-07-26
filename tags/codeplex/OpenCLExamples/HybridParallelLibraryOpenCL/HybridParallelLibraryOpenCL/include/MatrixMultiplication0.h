#ifndef MATRIXMULTIPLICATION0
#define MATRIXMULTIPLICATION0

#include "MatrixMultiplicationBase.h"
#include <string>

class MatrixMultiplication0 : public MatrixMultiplicationBase {

public:

	MatrixMultiplication0();

	~MatrixMultiplication0();

	virtual std::string getName() const;


protected:

	virtual void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	virtual void performAlgorithm();

	const char *algorithm() const;
};

#endif
