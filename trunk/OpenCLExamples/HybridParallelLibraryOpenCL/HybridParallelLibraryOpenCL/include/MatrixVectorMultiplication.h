#ifndef MATRIXVECTORMULTIPLICATION
#define MATRIXVECTORMULTIPLICATION

#include "ExampleBase.h"
#include <string>

class MatrixVectorMultiplication : public ExampleBase {

public:

	MatrixVectorMultiplication();

	~MatrixVectorMultiplication();

	virtual std::string getName() const;


protected:

    struct SpMat {
        size_t nrow;	//Num of rows;   
        size_t ncol;	//Num of columns;   
        size_t nnze;	//Num of non-zero elements;   
        float *val;		//Stores the non-zero elements;   
        size_t *rc_ind;	//Stores the row (CRS) or column (CCS) indexes of the elements in the val vector;   
        size_t *rc_ptr;	//Stores the locations in the val vector that start a row (CRS) or column (CCS);   
        size_t *rc_len;	//Stores the length of each column (CRS) or row (CCS);   
    };

    cl_mem matrixVals;
	cl_mem matrixRcIndex;
	cl_mem matrixRcPointer;
	cl_mem matrixNumRows;
	cl_mem inputVector;
	cl_mem resultVector;

	SpMat *sparseInputMatrix;
	float *fullInputMatrix;
	float *fullInputVector;

	static const float validationThreshold;

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void initializeInputMatrix();

	void initializeInputVector();

	void convertToCsr();

	void performAlgorithm();

	const char *algorithm() const;

	const bool isValid() const;

	const bool validateResult(float const* const& fullResultVector) const;

	const float* const matrixVectorMultiply() const;
};

#endif