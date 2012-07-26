//Sources:
// [1] http://code.google.com/p/thrust/wiki/QuickStartGuide
// [2] http://docs.thrust.googlecode.com/hg/group__transformed__reductions.html

//Usage:
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
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <thrust/inner_product.h>


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

bool innerProduct()
{
	thrust::host_vector<unsigned int> a(numElements);
	thrust::host_vector<unsigned int> b(numElements);

	thrust::sequence(a.begin(), a.end());
	thrust::sequence(b.begin(), b.end());
	
	thrust::device_vector<int> aa = a;
	thrust::device_vector<int> bb = b;

	for (int i = 0; i < numWarmupRounds; i++)
	{
		thrust::inner_product(aa.begin(), aa.end(), bb.begin(), 0);
	}

	float overallTime = 0;

	for (int i = 0; i < numMeasuredRounds; i++)
	{
		cudaEvent_t start_event, stop_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);

		cudaEventRecord(start_event, 0);

		result = thrust::inner_product(aa.begin(), aa.end(), bb.begin(), 0);

		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		
		float time = 0.0f;
		cudaEventElapsedTime(&time, start_event, stop_event);
		shrLog("The elapsed time for round %d is: %f ms.\n", i+1, time);
		overallTime += time;
	}

	for (int i = 0; i < numCoolDownRounds; i++)
	{
		thrust::inner_product(aa.begin(), aa.end(), bb.begin(), 0);
	}
	
	overallTime /= numMeasuredRounds;	

	shrLog("The result is: %u.\n", result);
	shrLog("The average elapsed time is: %f ms.\n\n", overallTime);
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
	bTestResult = innerProduct();

	bool bbbb = cutCheckCmdLineFlag( argc, (const char**)argv, "float");
	
    shrQAFinishExit(argc, (const char **)argv, bTestResult ? QA_PASSED : QA_FAILED);
}

