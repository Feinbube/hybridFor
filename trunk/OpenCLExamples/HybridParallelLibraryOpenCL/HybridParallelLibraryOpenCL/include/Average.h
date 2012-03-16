#ifndef AVERAGE
#define AVERAGE

#include "ExampleBase.h"
#include <iostream>
#include <string>
using namespace std;

class Average : public ExampleBase {
	public:
		Average();
		~Average();
		virtual std::string getName() const;

	protected:
		cl_mem a[MAX_GPU_COUNT];
		cl_mem b[MAX_GPU_COUNT];
		cl_mem c[MAX_GPU_COUNT];
		cl_mem len[MAX_GPU_COUNT];

		float *aArray;
		float *bArray;
		static const float validationThreshold;
		void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);
		void discardMembers();
		void initializeMembers();
		void performAlgorithm();
		const char *algorithm() const;
		const bool isValid() const;
};

#endif
