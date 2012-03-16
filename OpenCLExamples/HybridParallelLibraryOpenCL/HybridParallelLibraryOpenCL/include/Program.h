#ifndef PROGRAM
#define PROGRAM

#include <vector>
#include <string>
#include "examples.h"

using namespace std;

class Program {

public:

	Program();

	~Program();

    void RunBenchmark();


private:

    float sizeX;
    float sizeY;
    float sizeZ;

    int rounds;
    int warmup_rounds;
		

	static vector<ExampleBase *> *examples() 	{
		vector<ExampleBase *> *examples = new vector<ExampleBase *>();

//		examples->push_back(new Dummy());
//		examples->push_back(new Average());
//		examples->push_back(new SummingVectors());
		examples->push_back(new Ripple());
//		examples->push_back(new DotProduct());
//		examples->push_back(new HeatTransfer());
		// compiles but cannot be executed on current platform due to the lack of support for base atomics
		// examples->push_back(new Histogram());
		//examples->push_back(new RayTracing());
//		examples->push_back(new JuliaSet());

//		examples->push_back(new MatrixMultiplication0());
//		examples->push_back(new MatrixMultiplication1());
//		examples->push_back(new MatrixMultiplication2());
//		examples->push_back(new MatrixMultiplication3());
//		examples->push_back(new MatrixMultiplication4());
//		examples->push_back(new MatrixMultiplication5());
/*		examples->push_back(new Convolution());
		examples->push_back(new MinimumSpanningTree());
		examples->push_back(new MatrixVectorMultiplication());
		examples->push_back(new PrefixScan());
*/		//examples->push_back(new QuickSort());

		return examples;
	}

    void Benchmark(float size);

    void runExamples() const;

    void runExample(ExampleBase *example) const;
};

#endif
