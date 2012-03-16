#include "../include/Program.h"

Program::Program()
	:	rounds(20),
		warmup_rounds(5) { }

Program::~Program() { }

void Program::RunBenchmark() {
#ifndef DEBUG
    // disable additional benchmark runs during debug mode 
//    this->Benchmark(0.1f);
//    this->Benchmark(0.5f);
#endif
    this->Benchmark(1.0f);
}

void Program::Benchmark(float size) {
    this->sizeX = size;
    this->sizeY = size;
    this->sizeZ = size;

    this->runExamples();
}

void Program::runExamples() const {
	vector<ExampleBase *> *examples = Program::examples();

	for(vector<ExampleBase *>::iterator example = examples->begin(); example != examples->end(); ++example) 	{
		this->runExample(*example);
		delete *example;
	}

	delete examples;
}

void Program::runExample(ExampleBase *example) const {
    example->Run(
		this->sizeX,
		this->sizeY,
		this->sizeZ,
		this->rounds,
		this->warmup_rounds
	);
}
