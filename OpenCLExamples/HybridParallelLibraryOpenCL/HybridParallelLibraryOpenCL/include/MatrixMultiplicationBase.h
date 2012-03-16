#ifndef MATRIXMULTIPLICATIONBASE
#define MATRIXMULTIPLICATIONBASE

#include <cmath>
#include "ExampleBase.h"

class MatrixMultiplicationBase : public ExampleBase {
	public:
		MatrixMultiplicationBase();
		~MatrixMultiplicationBase();

	protected:
		cl_mem matrixA[MAX_GPU_COUNT];
		cl_mem matrixB;
		cl_mem matrixC[MAX_GPU_COUNT];
		cl_mem memSizeX[MAX_GPU_COUNT];
		cl_mem memSizeY;
		cl_mem memSizeZ;

		static const float validationThreshold;
		virtual void discardMembers();
		virtual void initializeMembers();
		virtual const bool isValid() const;
		const bool validateResult(float const* const& a, float const* const& b, float const* const& c) const;
};

#endif
