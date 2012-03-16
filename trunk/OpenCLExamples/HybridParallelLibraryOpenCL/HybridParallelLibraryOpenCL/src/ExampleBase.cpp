#include "../include/ExampleBase.h"

ExampleBase::ExampleBase()
	:	sizeX(0),
		sizeY(0),
		sizeZ(0),
		platform_id(NULL),
		context(NULL), 
		program(NULL),
		deviceCount(0) { }

ExampleBase::~ExampleBase() {
    clReleaseProgram(this->program);
    clReleaseContext(this->context);

	free(device_ids);
	releaseKernels(this->kernel);

	for (unsigned int i=0; i<deviceCount; i++) {
		if (commands[i])
			clReleaseCommandQueue(commands[i]);
	}
}

void ExampleBase::setup() {
	this->discardMembers();
	this->initializeOpenCL();
	this->initializeMembers();
}

void ExampleBase::Run(float sizeX, float sizeY, float sizeZ, int rounds, int warmupRounds) {
	clock_t start, end;

	cout << "Running " << this->getName() << endl;

    this->scaleAndSetSizes(sizeX, sizeY, sizeZ);

    cout << "(" << this->sizeX << " / " << this->sizeY << " / " << this->sizeZ << ")..." << endl;
            
    this->setup();
    this->runTimes(warmupRounds);
	
	start = clock();
    this->runTimes(rounds);
	end = clock();

    this->runTimes(warmupRounds);

    cout << "Done in " << (end - start) * 1000 / CLOCKS_PER_SEC << "ms. " << (this->isValid() ? "SUCCESS" : "<!!! FAILED !!!>") << endl;
}

const bool ExampleBase::supportsBaseAtomics() const {
	const size_t bufferSize = 2048;
	char buffer[bufferSize];

	int error = clGetPlatformInfo(
		this->platform_id,
		CL_PLATFORM_EXTENSIONS,
		bufferSize,
		buffer,
		NULL);
    if (error != CL_SUCCESS)
        this->exitWith("Error: Failed to retrieve platform extension info for " + this->getName() + "!", error);

	if (strstr(buffer, "cl_khr_global_int32_base_atomics"))
		return true;
	else
		return false;
}

void ExampleBase::runTimes(int numberOfTimes) {
    for(int warmupRound = 0; warmupRound < numberOfTimes; warmupRound++)
        this->performAlgorithm();
}

const float ExampleBase::randomFloat() const {
	return (((float) rand() / RAND_MAX) - 0.5f) * 2;
}

const size_t ExampleBase::localWorkSize(const size_t maxLocalWorkSize, const int numberOfWorkItems, const int dimensions) const {
	size_t localWorkSize = (size_t) pow((float) maxLocalWorkSize, (float) 1/dimensions);
	while ((localWorkSize /= 2) > (size_t) numberOfWorkItems) continue;
	return localWorkSize * 2;
}

const size_t ExampleBase::globalWorkSize(const size_t localWorkSize, const int numberOfWorkItems) const {
	size_t globalWorkSize = 0;

	while (globalWorkSize < (size_t) numberOfWorkItems/deviceCount)
		globalWorkSize += localWorkSize;
		
	return globalWorkSize;
}

void ExampleBase::initializeOpenCL() {
    int err;

	cl_platform_id platform_ids[10];

	err = clGetPlatformIDs(10, platform_ids, NULL);
    if (err != CL_SUCCESS)
        this->exitWith("Error: Failed get platoform id for " + this->getName() + "!", err);

	for(int i=0; i<10; i++)
	{
		platform_id = platform_ids[i];
		err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &this->deviceCount);		// get device count
		if(deviceCount > 0)
			break;
	}


	this->device_ids = (cl_device_id *)malloc(deviceCount * sizeof(cl_device_id));			// allocate memory for device_ids array
	// TODO fehler bei deviceCount > MAX_GPU_COUNT
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, deviceCount, device_ids, NULL);	// get device ids
    if (err != CL_SUCCESS)
        this->exitWith("Error: Failed to create a device group for " + this->getName() + "!", err);

cout << "using " << deviceCount << " devices" << endl;	


	// TODO mehrere Kontexte vergleichen.. maybe
	this->context = clCreateContext(0, deviceCount, device_ids, NULL, NULL, &err);
    if (!this->context)
        this->exitWith("Error: Failed to create a compute context for " + this->getName() + "!", err);

	for (unsigned int i=0; i<deviceCount; i++) {
		this->commands[i] = clCreateCommandQueue(this->context, device_ids[i], 0, &err);
	
		if (!this->commands)
			this->exitWith("Error: Failed to create a command commands for " + this->getName() + "!", err);
	}
		
	const char *algo = this->algorithm();
	this->program = clCreateProgramWithSource(this->context, 1, (const char **) &algo, NULL, &err);
    if (!this->program)
        this->exitWith("Error: Failed to create compute program for " + this->getName() + "!", err);

    err = clBuildProgram(this->program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)     {
        size_t len;
        char buffer[2048];

        clGetProgramBuildInfo(this->program, this->device_ids[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        this->exitWith("Error: Failed to build program executable for " + this->getName() + "!", err);
    }

	createKernels(this->kernel, this->program);
	/*
	for (int i=0; i<deviceCount; i++) {
		this->kernel[i] = clCreateKernel(this->program, "algorithm", &err);
		if (!this->kernel || err != CL_SUCCESS)
			this->exitWith("Error: Failed to create compute kernel for " + this->getName() + "!", err);
	}*/	
}

void ExampleBase::exitWith(string message, int error) const {
	cout << message << endl;
	exit(error);
}

void ExampleBase::printField(float const* const& field, int maxX) const {
	cout << "| ";
	for(int x = 0; x < maxX; x++)
		cout << field[x] << (x != (maxX - 1) ? " | " : " |");
}

void ExampleBase::printField(float const* const& field, int maxX, int maxY) const {
	for(int y = 0; y < maxY; y++) 	{
		this->printField(&field[y * maxX], maxX);
		cout << endl;
	}
}
