# Hybrid.Net
Hybrid.Net enables .NET developers to harness the power of GPUs for data- and compute-intense applications using the simple well-known construct: Parallel.For

Hybrid.Net brings to you:
* High-performance with state-of-the-art accelerators!
* A light-weight well-known programming model!
* As a library that can be easily added to your project! (no precompiler. no external tool-chain.)

## Example - Let's see how it works.
The following example of a vector add demonstrates how Hybrid.Net is used. There are three vectors a, b and c of the same size and these vectors are to be added to each other in the way c = a + b. We have Hybrid.Net performing this function in parallel on the GPU using the parallel for construct:

```
//define vectors a, b and c
int vectorLength = 1024;
int[] a = new int[vectorLength];
int[] b = new int[vectorLength];
int[] c = new int[vectorLength];

//initialize vectors a, b and c
…

//execute vector addition on GPU
Hybrid.Parallel.For(Execute.OnSingleGpu, 0, arrayLength, delegate(int i)
{
    c[i] = a[i] + b[i];
});
```

Hybrid.Net automatically takes care to move the vectors between main memory and GPU memory. This is done implicitly as the data is used within the Hybrid for-loop.
In order to provide a light-weight interface to parallel GPU computing, the for-loop construct of Hybrid.Net is similar to [the one that ships with .NET Framework 4.0](http://msdn.microsoft.com/en-us/library/system.threading.tasks.parallel.aspx). It takes the same arguments, as well as an (optional) indicator of the platform that is to be used to execute the loop body.

## What's in it for me?
### Features
The Hybrid.Net prototype offers a parallel for-loop with the following features:
 * An API that is similar to the System.Threading.Tasks.Parallel.For API plus an additional argument to specify the preferred execution mode. Available execution modes are:
   * OnSingleGPU
   * OnAllGPUs (currently not supported)
   * OnSingleCPU
   * OnAllCPUs
   * OnEverythingAvailable (currently not supported)
 * In addition there is a loop that can be used to replace two iterators and efficiently and conveniently operate on 2-dimensional arrays.
 * Multi-dimensional arrays are supported.
 * Even though methods of other objects are not supported, the following functions can be used:
   * Random functions (e.g. Random.Next(), Random.NextDouble(), Random.NextBytes(), Random.NextUInt())
   * Mathematical functions (e.g. Math.Max(,), Math.Min(,), Math.Abs(,), …)
   * Calls to methods of the object in which the Hybrid.Net for-loop resides

### Restrictions
Since Hybrid.Net is a prototype, there are still some features it does not support:
 * Hybrid.Net does not support the handling of objects in the loop body. One can only use primitive data types and arrays of primitive data types. Exceptions are stated in the feature section (Random and Math)
 * Since graphics cards do not support recursion, Hybrid.Net also does not support recursions
 * ATI graphics cards are currently not supported, because they do not support "Goto" which the current version of our code generator relies on.

## Download and Installation
### Binary Distribution
In the downloads section, there are binary distributions of the Hybrid.Net library for the x86 and the x64 architecture. Simply download and extract the .zip-file and add the Hybrid.Net library to your project. (More details about adding external libraries are available at [HowTo](http://msdn.microsoft.com/en-us/library/wkze6zky%28v=vs.100%29.aspx).)
### Examples
The source code package includes several examples on how to use the special for-loop provided by Hybrid.Net. These examples include but are not limited to different matrix multiplication algorithms, a convolution filter and a quicksort implementation. If you want to try them follow the instructions to download the source code and see the Benchmarks and Tests (next section) for more information.

## Benchmarks and Tests
The Hybrid.Net prototype source code contains not only the sources but also a test suite and a benchmark suite.
###Test Suite
The test suite is a separate project in the Hybrid.Net solution. It is called Hybrid.Tests and contains all the examples from the ```Hybrid.Examples``` project. These tests can be selected and run as common unit tests in Visual Studio. (For help on running unit tests see this [Walkthrough](http://msdn.microsoft.com/en-us/library/ms182532.aspx).) Note that you have to adjust the test host depending on the target CPU architecture. (See this [HowTo](http://msdn.microsoft.com/en-us/library/ee782531) on setting the test host.)
###Benchmark Suite
The benchmark suite is represented by the Hybrid.Benchmark project within the Hybrid solution. It contains a file called ```Program.cs``` in which you can define which benchmarks you want to run. You can use any of the examples in the ```Hybrid.Examples``` project to benchmark the Hybrid prototype. Simply add the examples to or remove the them from list returned by the ```static List<ExampleBase> examples()``` method.
The benchmark suite also supports defining the number of warmup rounds and rounds to be measured. This can be done in the Program.cs file inside the ```private void benchmark(...)``` method. The number of warmup rounds is also taken as the number of cool-down rounds. 
Some of the examples supply a print out on the command line. To enable this print out set the ```print``` variable in the ```private void benchmark(...)``` method to ```true``` (default is ```false```).

Note that this benchmark suite is not intended to represent a special real life workload, it is rather designed for performance comparisons between the GPU and the CPU executions for each of the provided examples.

## General Notes
The Hybrid library relies on the [OpenCL.Net project by Illusio](http://sourceforge.net/projects/openclnet/). OpenCL.Net is an OpenCL wrapper for .NET. We use it to access the native OpenCL implementation.
