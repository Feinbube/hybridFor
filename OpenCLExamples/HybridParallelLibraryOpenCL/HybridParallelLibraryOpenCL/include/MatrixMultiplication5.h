#ifndef MATRIXMULTIPLICATION5
#define MATRIXMULTIPLICATION5

#include "MatrixMultiplicationBase.h"
#include <string>

class MatrixMultiplication5 : public MatrixMultiplicationBase {

public:

	MatrixMultiplication5();

	~MatrixMultiplication5();

	virtual std::string getName() const;


protected:
	
	cl_mem mf;
	cl_mem ml;
	cl_mem nf;
	cl_mem nl;
	cl_mem pf;
	cl_mem pl;

	void discardMembers();

	void initializeMembers();

	virtual void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	virtual void performAlgorithm();

	const char *algorithm() const;


private:
	
	size_t local[2];
	size_t global[2];

	virtual void setRanges(int mf, int ml, int nf, int nl, int pf, int pl) const;

	void matmultrec(int mf, int ml, int nf, int nl, int pf, int pl);

    void matmultleaf(int mf, int ml, int nf, int nl, int pf, int pl);
};

#endif