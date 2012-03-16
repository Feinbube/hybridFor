#ifndef RIPPLE
#define RIPPLE

#include "ExampleBase.h"
#include <string>

class Ripple : public ExampleBase {
	public:
		Ripple();
		~Ripple();
		virtual std::string getName() const;

	protected:
		cl_mem bitmap[MAX_GPU_COUNT];
		cl_mem memSizeX;
		cl_mem memSizeY;
		cl_mem offset[MAX_GPU_COUNT];

		static const float validationThreshold;
		void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);
		void discardMembers();
		void initializeMembers();
		void performAlgorithm();
		const char *algorithm() const;
		const bool isValid() const;
};

#endif
