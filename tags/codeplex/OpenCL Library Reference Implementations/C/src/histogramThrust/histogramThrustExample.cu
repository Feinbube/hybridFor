#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <cutil_inline.h>
#include <iostream>
#include <iomanip>
#include <iterator>

// This example illustrates several methods for computing a
// histogram [1] with Thrust.  We consider standard "dense"
// histograms, where some bins may have zero entries, as well
// as "sparse" histograms, where only the nonzero bins are
// stored.  For example, histograms for the data set
//    [2 1 0 0 2 2 1 1 1 1 4]
// which contains 2 zeros, 5 ones, and 3 twos and 1 four, is
//    [2 5 3 0 1]
// using the dense method and 
//    [(0,2), (1,5), (2,3), (4,1)]
// using the sparse method. Since there are no threes, the 
// sparse histogram representation does not contain a bin
// for that value.
//
// Note that we choose to store the sparse histogram in two
// separate arrays, one array of keys and one array of bin counts,
//    [0 1 2 4] - keys
//    [2 5 3 1] - bin counts
// This "structure of arrays" format is generally faster and
// more convenient to process than the alternative "array
// of structures" layout.
//
// The best histogramming methods depends on the application.
// If the number of bins is relatively small compared to the 
// input size, then the binary search-based dense histogram
// method is probably best.  If the number of bins is comparable
// to the input size, then the reduce_by_key-based sparse method 
// ought to be faster.  When in doubt, try both and see which
// is fastest.
//

// Sources:
// [1] http://en.wikipedia.org/wiki/Histogram
// [2] http://code.google.com/p/thrust/downloads/detail?name=examples-1.6.zip&can=2&q=  --> sample: histogram.cu

// Usage:
// Command line parameters:
	// n : number of input elements
	// w : number of warmup rounds
	// m : number of measured rounds
	// c : number of cooldown rounds
	
	// example: -m=10 => 10 rounds with measured time are executed

int n = 1048576;
int s = 4;
int numWarmupRounds = 5;
int numMeasuredRounds = 3;
int numCoolDownRounds = 2;


// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

// dense histogram using binary search
template <typename Vector1, 
          typename Vector2>
float dense_histogram(const Vector1& input,
                           Vector2& histogram)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);

  //start measuring time
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);


  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = 256 + 1;
  
  // find the end of each bin of values to create cumulative histogram
  thrust::counting_iterator<IndexType> search_begin(0);
  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());

  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());

  //stop measuring time
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);

  float time = 0.0f;
  cudaEventElapsedTime(&time, start_event, stop_event);

  // print the histogram
  //print_vector("histogram", histogram);

  return time;
}

void processCommandLine(int argc, char **argv)
{
	int cmdVali = 0;
	if (cutGetCmdLineArgumenti( argc, (const char**)argv, "n", &cmdVali))
	{
		n = cmdVali;
		printf("The number of elements is set to: %d\n", n);
	}
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "w", &cmdVali))
	{
		numWarmupRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numWarmupRounds);
	}
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "m", &cmdVali))
	{
		numMeasuredRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numMeasuredRounds);
	} 
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "c", &cmdVali))
	{
		numCoolDownRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numCoolDownRounds);
	}
}

int main(int argc, char **argv)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 256);

  processCommandLine(argc, argv);

  // generate random data on the host
  thrust::host_vector<int> input(n);
  for(int i = 0; i < n; i++)
  {
    int sum = 0;
    for (int j = 0; j < s; j++)
      sum += dist(rng);
    input[i] = sum / s;
  }

  thrust::device_vector<int> histogram(256 + 1);
  
  for (int i = 0; i < numWarmupRounds; i++)
  {
	dense_histogram(input, histogram);
  }

  float overallTime = 0.0f;
  for (int i = 0; i < numMeasuredRounds; i++)
  {
	float time = dense_histogram(input, histogram);
	printf("The elapsed time for round %d is: %f ms.\n", i+1, time);
	overallTime += time;
  }

  for (int i = 0; i < numCoolDownRounds; i++)
  {
	dense_histogram(input, histogram);
  }

  overallTime /= numMeasuredRounds;
  printf("The average elapsed time is: %f ms.\n\n", overallTime);
  return 0;
}

