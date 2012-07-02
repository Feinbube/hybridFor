//Sources:
// [1] http://developer.nvidia.com/cuda-cc-sdk-code-samples --> sample: CUDA Radix Sort using the Thrust Library
// [2] http://code.google.com/p/thrust/wiki/QuickStartGuide
// [3] http://code.google.com/p/thrust/downloads/detail?name=examples-1.6.zip&can=2&q= --> sample: arbitrary_transformation.cu

// Command line parameters:
	// n : number of input elements per Vector
	// w : number of warmup rounds
	// m : number of measured rounds
	// c : number of cooldown rounds
	
	// example: -m=10 => 10 rounds with measured time are executed


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

#include <cutil_inline.h>


#include <shrUtils.h>
#include <shrQATest.h>
#include <algorithm>
#include <time.h>
#include <limits.h>

int numWarmupRounds = 5;
int numMeasuredRounds = 3;
int numCoolDownRounds = 2;
int numElements = 1048576; 
unsigned int result;

struct average_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // C[i] = (A[i] + B[i]) / 2;
        thrust::get<2>(t) = (thrust::get<0>(t) + thrust::get<1>(t)) >> 1;
    }
};


bool vectorAverage()
{
	thrust::host_vector<int> a(numElements);
	thrust::host_vector<int> b(numElements);
	thrust::host_vector<int> h_result(numElements, 0);
	
	thrust::generate(a.begin(), a.end(), rand);
	thrust::generate(b.begin(), b.end(), rand);

	thrust::device_vector<int> aa = a;
	thrust::device_vector<int> bb = b;
	thrust::device_vector<int> d_result(numElements, 0);

	for (int i = 0; i < numWarmupRounds; i++)
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(aa.begin(), bb.begin(), d_result.begin())),
						 thrust::make_zip_iterator(thrust::make_tuple(aa.end(), bb.end(), d_result.end())),
						 average_functor());
	}

	float overallTime = 0;

	for (int i = 0; i < numMeasuredRounds; i++)
	{
		cudaEvent_t start_event, stop_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);

		cudaEventRecord(start_event, 0);

		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(aa.begin(), bb.begin(), d_result.begin())),
						 thrust::make_zip_iterator(thrust::make_tuple(aa.end(), bb.end(), d_result.end())),
						 average_functor());

		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		
		float time = 0.0f;
		cudaEventElapsedTime(&time, start_event, stop_event);
		shrLog("The elapsed time for round %d is: %f ms.\n", i+1, time);
		overallTime += time;
	}

	for (int i = 0; i < numCoolDownRounds; i++)
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(aa.begin(), bb.begin(), d_result.begin())),
						 thrust::make_zip_iterator(thrust::make_tuple(aa.end(), bb.end(), d_result.end())),
						 average_functor());
	}
	
	overallTime /= numMeasuredRounds;	

	shrLog("The result is: %u.\n", result);
	shrLog("The average elapsed time is: %f ms.\n\n", overallTime);

	h_result = d_result;
	/*
	for(int i = 0; i < 20; i++)
	{
        std::cout << "(" << a[i] << " + " << b[i] << ")/ 2 =" << h_result[i] << std::endl;
	}
	*/

    return true;
}

void processCommandLine(int argc, char **argv)
{
	int cmdVali = 0;
	if (cutGetCmdLineArgumenti( argc, (const char**)argv, "n", &cmdVali))
	{
		numElements = cmdVali;
		shrLog("The number of elements is set to: %d\n", numElements);
	}
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "w", &cmdVali))
	{
		numWarmupRounds = cmdVali;
		shrLog("The number of warm-up rounds is set to: %d\n", numWarmupRounds);
	}
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "m", &cmdVali))
	{
		numMeasuredRounds = cmdVali;
		shrLog("The number of warm-up rounds is set to: %d\n", numMeasuredRounds);
	} 
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "c", &cmdVali))
	{
		numCoolDownRounds = cmdVali;
		shrLog("The number of warm-up rounds is set to: %d\n", numCoolDownRounds);
	}
}

int main(int argc, char **argv)
{
    shrQAStart(argc, argv);

    // Start logs
    shrSetLogFileName ("radixSort.txt");
    shrLog("%s Starting...\n\n", argv[0]);
    
    cutilDeviceInit(argc, argv);

	processCommandLine(argc, argv);
  
    bool bTestResult = false;
	bTestResult = vectorAverage();

	bool bbbb = cutCheckCmdLineFlag( argc, (const char**)argv, "float");
	
    shrQAFinishExit(argc, (const char **)argv, bTestResult ? QA_PASSED : QA_FAILED);
}

