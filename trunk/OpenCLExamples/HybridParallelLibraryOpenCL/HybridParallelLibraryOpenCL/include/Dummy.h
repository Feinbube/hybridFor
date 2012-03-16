#ifndef DUMMY
#define DUMMY

#include "ExampleBase.h"
#include <iostream>
#include <string>
using namespace std;

class Dummy : public ExampleBase {
	public:
		Dummy();
		~Dummy();
		virtual std::string getName() const;
	protected:
		cl_mem in[MAX_GPU_COUNT];
		cl_mem out[MAX_GPU_COUNT];
		cl_mem len[MAX_GPU_COUNT];
	
		int sizePerDevice;
		int workSize[MAX_GPU_COUNT];
		int workOffset[MAX_GPU_COUNT];
		float *inArray;
		float *outArray;
		void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);
		void discardMembers();
		void initializeMembers();
		void performAlgorithm();
		const char *algorithm() const;
		const bool isValid() const;
};

#endif
