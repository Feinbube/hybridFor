using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using Hybrid.Examples;
using Hybrid.Examples.CudaByExample;
using Hybrid.Examples.Upcrc2010;
using Hybrid.Examples.Upcrc2010.MatrixMultiplication;

namespace Hybrid.Benchmark
{
    public class Program
    {
        double sizeX;
        double sizeY;
        double sizeZ;

        int rounds;
        int warmup_rounds;

        bool print;

        TextWriter tw;

        static void Main(string[] args)
        {
            Environment.SetEnvironmentVariable("CL_LOG_ERRORS", "stdout");

            new Program().RunBenchmark();

            Console.WriteLine("Press ENTER to quit...");
            Console.ReadLine();
        }

        static TextWriter evaluationLog()
        {
            string fileName = "ForGPU_Evaluation_"
                + DateTime.Now.ToShortDateString() + "_"
                + DateTime.Now.ToShortTimeString().Replace(":", ".") + "."
                + DateTime.Now.Second
                + ".csv";

            TextWriter tw = File.CreateText(fileName);

            tw.WriteLine("alogrithm;scaleX;scaleY;scaleZ;mode;executionTime;valid");

            return tw;
        }

        static List<ExampleBase> examples()
        {
            return new List<ExampleBase>(new ExampleBase[]{
            new MatrixMultiplication0(),
            new MatrixMultiplication1(),
            new MatrixMultiplication2(),
            new MatrixMultiplication3(),
            new MatrixMultiplication4(),
            new MatrixMultiplication5(),

            new Convolution(),
            //new MatrixVectorMultiplication(),
            new MinimumSpanningTree(),
            new PrefixScan(),
            new QuickSort(),

            new Average(),
            new DotProduct(),
            new HeatTransfer(),
            new Histogram(),
            new JuliaSet(),
            //new RayTracing(),
            new Ripple(),
            new SummingVectors()
            });
        }

        private void RunBenchmark()
        {
            Benchmark(0.1);
            Benchmark(0.5);
            Benchmark(1.0);
        }

        private void Benchmark(double size)
        {
            tw = evaluationLog();

            rounds = 20;
            warmup_rounds = 5;

            print = false;

            sizeX = size;
            sizeY = size;
            sizeZ = size;

            runExamples();

            tw.Close();
        }

        private void RunFindGoodSizes()
        {
            foreach (ExampleBase example in examples())
                findGoodSize(example);
        }

        private void findGoodSize(ExampleBase example)
        {
            double executionTime = 0.9;
            double scale = 0.0;

            while (executionTime < 1.0)
            {
                scale += 15.0 - executionTime * 10.0;

                Parallel.Mode = ExecutionMode.Gpu;
                Atomic.Mode = ExecutionMode.Gpu;

                executionTime = example.Run(scale, scale, scale, false, 20, 5, null);

                System.Threading.Thread.Sleep(100);
            }

            Console.WriteLine("Scale " + scale + " for " + example.GetType().Name + " for " + executionTime + "s.");
        }


        private void runExamples()
        {
            foreach (ExampleBase example in examples())
                runExample(example);
        }

        private void runExample(ExampleBase forGpuExample)
        {
            Console.Write("[Serial]     ");
            runExample(forGpuExample, ExecutionMode.Serial);

            Console.Write("[GPU]        ");
            runExample(forGpuExample, ExecutionMode.Gpu);
            Parallel.ReInitialize();

            Console.Write("[Parallel]   ");
            runExample(forGpuExample, ExecutionMode.TaskParallel);

            Console.Write("[Parallel2D] ");
            runExample(forGpuExample, ExecutionMode.TaskParallel2D);

            Console.Write("[GPU2D]      ");
            runExample(forGpuExample, ExecutionMode.Gpu2D);
            Parallel.ReInitialize();

            Console.WriteLine();
        }

        private void runExample(ExampleBase forGpuExample, ExecutionMode mode)
        {
            Parallel.Mode = mode;
            Atomic.Mode = mode;

            forGpuExample.Run(sizeX, sizeY, sizeZ, print, rounds, warmup_rounds, tw);

            System.Threading.Thread.Sleep(500);
        }
    }
}
