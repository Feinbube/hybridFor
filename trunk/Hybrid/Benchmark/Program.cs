using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

using Hybrid.Examples;
using Hybrid.Examples.CudaByExample;
using Hybrid.Examples.Upcrc2010;
using Hybrid.Examples.Upcrc2010.MatrixMultiplication;
using System.Threading;
using System.Globalization;

namespace Hybrid.Benchmark
{
    public class Program
    {
        static SystemCharacteristics systemCharacteristics = new SystemCharacteristics();

        double minSequentialExecutionTime;

        int rounds;
        int warmup_rounds;

        bool print;

        TextWriter tw;

        static void Main(string[] args)
        {
            Environment.SetEnvironmentVariable("CL_LOG_ERRORS", "stdout");

            Console.WriteLine(systemCharacteristics.ToString());

            new Program().RunBenchmark();

            Console.WriteLine("Press ENTER to quit...");
            Console.ReadLine();
        }

        static TextWriter evaluationLog()
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("de-DE");

            string fileName = "Evaluation_"
                + System.Environment.MachineName + "_"
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
                
                /* // function tests
                new StaticFunctionCall(),
                new LocalFunctionCall(),
                new Lists(), */

                /* // big examples
                new Crypt3(),
                new GameOfLife(),
                new Wator(), */
                new SudokuValidator(),

                new MatrixMultiplication0(),
                new MatrixMultiplication1(),
                new MatrixMultiplication2(),
                new MatrixMultiplication3(),
                new MatrixMultiplication4(),
                new MatrixMultiplication5(),

                new Convolution(),
                
                /* // example is to small
                new MatrixVectorMultiplication(), */

                new MinimumSpanningTree(),
                new PrefixScan(),

                /* // not Gpu-enabled
                new QuickSort(), */

                new Average(),
                new DotProduct(),
                new HeatTransfer(),
                new Histogram(),
               
                /* // not reliable
                new JuliaSet(),*/

                new RayTracing(),
                new Ripple(),
                new SummingVectors()
            });
        }

        private void RunBenchmark()
        {
            benchmark(0.5);
            //benchmark(3.0);
            //benchmark(4.0);
        }

        private void benchmark(double minSequentialExecutionTime)
        {
            tw = evaluationLog();

            this.minSequentialExecutionTime = minSequentialExecutionTime;

            rounds = 20;
            warmup_rounds = 5;

            print = false;

            runExamples();

            tw.Close();
        }

        private void runExamples()
        {
            foreach (ExampleBase example in examples())
                runExample(example);
        }

        private void runExample(ExampleBase example)
        {
            double sizeFactor = systemCharacteristics.GetScale(example, minSequentialExecutionTime);

            Console.Write("[Automatic] ");
            runExample(example, Execute.OnEverythingAvailable, sizeFactor);
            Parallel.ReInitialize();

            Console.Write("[Serial]    ");
            runExample(example, Execute.OnSingleCpu, sizeFactor);

            if (!systemCharacteristics.Platform.ContainsAGpu)
                Console.WriteLine("[GPU]       No GPUs available!");
            else
            {
                Console.Write("[GPU]       ");
                runExample(example, Execute.OnSingleGpu, sizeFactor);
                Parallel.ReInitialize();
            }

            Console.Write("[Parallel]  ");
            runExample(example, Execute.OnAllCpus, sizeFactor);

            Console.WriteLine();
        }

        private void runExample(ExampleBase example, Execute mode, double sizeFactor)
        {
            example.ExecuteOn = mode;
            try
            {
                example.Run(sizeFactor, sizeFactor, sizeFactor, print, rounds, warmup_rounds, tw);
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception.ToString());
            }

            System.Threading.Thread.Sleep(100);
        }
    }
}
