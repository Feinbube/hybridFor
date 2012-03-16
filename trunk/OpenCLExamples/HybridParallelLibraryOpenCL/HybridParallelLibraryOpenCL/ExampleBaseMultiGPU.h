#ifndef EXAMPLEBASE
#define EXAMPLEBASE

#include <string>
#include <iostream>
#include <time.h>
#include <limits>
#include <CL/cl.h>

using namespace std;

class ExampleBase {

public:

	ExampleBase();

	~ExampleBase();

	virtual std::string getName() const = 0;

	void Run(float sizeX, float sizeY, float sizeZ, int rounds, int warmupRounds);


protected:

	cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;

    int sizeX;
    int sizeY;
    int sizeZ;

	const float randomFloat() const;
    
	virtual const bool isValid() const = 0;

	virtual void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ) = 0;
    
	void setup();

	virtual void discardMembers() = 0;

	virtual void initializeMembers() = 0;

	virtual void performAlgorithm() = 0;

	const bool supportsBaseAtomics() const;

	const size_t localWorkSize(const size_t maxLocalWorkSize, const int numberOfWorkItems, const int dimensions) const;

	const size_t globalWorkSize(const size_t localWorkSize, const int numberOfWorkItems) const;

	virtual const char *algorithm() const = 0;

	virtual void initializeOpenCL();

	void exitWith(string message, int error) const;

	void printField(float const* const& field, int maxX) const;

	void printField(float const* const& field, int maxX, int maxY) const;

	inline const int indexOf(const int x, const int y) const {
		return x * this->sizeX + y;
	}


private:

	void runTimes(int numberOfTimes);
};

#endif