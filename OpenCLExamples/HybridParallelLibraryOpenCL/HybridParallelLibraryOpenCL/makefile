CC=g++
CFLAGS=-c -DDEBUG -g -Wall
LDFLAGS=-lOpenCL
INC=-Icommon/inc -Ishared/inc -Iinclude
SOURCES=main.cpp src/PrefixScan.cpp src/MatrixMultiplication0.cpp src/RayTracing.cpp src/MatrixMultiplication1.cpp src/MatrixMultiplication2.cpp src/MatrixMultiplication3.cpp src/MatrixVectorMultiplication.cpp src/MatrixMultiplication4.cpp src/Program.cpp src/MatrixMultiplication5.cpp src/Average.cpp src/MatrixMultiplicationBase.cpp src/Convolution.cpp src/JuliaSet.cpp src/QuickSort.cpp src/SummingVectors.cpp src/HeatTransfer.cpp src/ExampleBase.cpp src/DotProduct.cpp src/MinimumSpanningTree.cpp src/Ripple.cpp src/Histogram.cpp src/Dummy.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(INC) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INC) $< -o $@

clean:
	rm -rf *.o src/*.o
