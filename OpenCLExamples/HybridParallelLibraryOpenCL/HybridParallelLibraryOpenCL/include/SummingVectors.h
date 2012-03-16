#ifndef SUMMINGVECTORS
#define SUMMINGVECTORS

#include "ExampleBase.h"
#include <string>

class SummingVectors : public ExampleBase {
	public:
		SummingVectors();
		~SummingVectors();
		virtual std::string getName() const;

	protected:
		cl_mem a[MAX_GPU_COUNT];
		cl_mem b[MAX_GPU_COUNT];
		cl_mem c[MAX_GPU_COUNT];
		cl_mem len[MAX_GPU_COUNT];

		size_t *aArray;
		size_t *bArray;
		void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);
		void discardMembers();
		void initializeMembers();
		void performAlgorithm();
		const char *algorithm() const;
		const bool isValid() const;
};

#endif
