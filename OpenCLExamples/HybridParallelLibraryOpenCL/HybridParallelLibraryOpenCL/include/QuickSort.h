#ifndef QUICKSORT
#define QUICKSORT

#include "ExampleBase.h"
#include <string>

class QuickSort : public ExampleBase {

public:

	QuickSort();

	~QuickSort();

	virtual std::string getName() const;


protected:

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void performAlgorithm();

	const char *algorithm() const;

	const bool isValid() const;
};

#endif