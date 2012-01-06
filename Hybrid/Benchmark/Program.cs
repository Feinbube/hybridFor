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

            tw.WriteLine("alogrithm;scaleX;scaleY;scaleZ;AbsSerial;AbsCPUs;AbsGPU;AbsAll;SpdUpSerial;SpdUpCPUs;SpdUpGPU;SpdUpAll;");

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
                new SudokuValidatorInvalidColumn(),
                new SudokuValidatorInvalidNumbers(),
                new SudokuValidatorInvalidRow(),
                new SudokuValidatorInvalidSubfield(),
                new SudokuValidator2D(),

                new MatrixMultiplication0(),
                new MatrixMultiplication1(),
                new MatrixMultiplication2(),
                new MatrixMultiplication3(),
                new MatrixMultiplication4(),
                new MatrixMultiplication5(),

                new Convolution(),
                new MatrixVectorMultiplication(),

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
            //benchmark(0.5);
            benchmark(3.0);
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

            Console.Write("[Serial]    ");
            ExampleBase.RunResult runResultSerial = runExample(example, Execute.OnSingleCpu, sizeFactor);

            Console.Write("[Parallel]  ");
            ExampleBase.RunResult runResultParallel = runExample(example, Execute.OnAllCpus, sizeFactor);

            Console.Write("[Automatic] ");
            ExampleBase.RunResult runResultAutomatic = runExample(example, Execute.OnEverythingAvailable, sizeFactor);
            Parallel.ReInitialize();

            if(runResultAutomatic == null)
                runResultAutomatic = new ExampleBase.RunResult() { Valid = false, ElapsedTotalSeconds = -1 };
            

            ExampleBase.RunResult runResultGPU = null;
            if (!systemCharacteristics.Platform.ContainsAGpu)
                Console.WriteLine("[GPU]       No GPUs available!");
            else
            {
                Console.Write("[GPU]       ");
                runResultGPU = runExample(example, Execute.OnSingleGpu, sizeFactor);
                Parallel.ReInitialize();
            }

            if (runResultGPU == null)
                runResultGPU = new ExampleBase.RunResult() { Valid = false, ElapsedTotalSeconds = -1 };
            
            Console.WriteLine();

            writeOutputs(example, runResultSerial, runResultParallel, runResultGPU, runResultAutomatic);
        }

        private void writeOutputs(ExampleBase example, ExampleBase.RunResult runResultSerial, ExampleBase.RunResult runResultParallel, ExampleBase.RunResult runResultGPU, ExampleBase.RunResult runResultAutomatic)
        {
            reasonablyEqual(runResultSerial, runResultParallel);
            reasonablyEqual(runResultSerial, runResultGPU);
            reasonablyEqual(runResultSerial, runResultAutomatic);

            double relSerial = runResultSerial.RelativeExecutionTime(runResultSerial.ElapsedTotalSeconds);
            double relCPUs = runResultParallel.RelativeExecutionTime(runResultSerial.ElapsedTotalSeconds);
            double relGPU = runResultGPU.RelativeExecutionTime(runResultSerial.ElapsedTotalSeconds);
            double relAll = runResultAutomatic.RelativeExecutionTime(runResultSerial.ElapsedTotalSeconds);

            tw.WriteLine(example.Name + ";" + runResultSerial.SizeX + ";" + runResultSerial.SizeY + ";" + runResultSerial.SizeZ + ";"
                + runResultSerial.ElapsedTotalSeconds + ";" + runResultParallel.ElapsedTotalSeconds + ";" + runResultGPU.ElapsedTotalSeconds + ";" + runResultAutomatic.ElapsedTotalSeconds + ";"
                + relSerial + ";" + relCPUs + ";" + relGPU + ";" + relAll + ";");
            tw.Flush();
        }

        private void reasonablyEqual(ExampleBase.RunResult one, ExampleBase.RunResult other)
        {
            if (!other.Valid)
                return;

            if (!one.Valid)
            {
                Console.WriteLine("!!!!!!!!!!!!!!!!!!!!!!");
                Console.WriteLine("!!!!ONE IS INVALID!!!!");
                Console.WriteLine("!!!!!!!!!!!!!!!!!!!!!!");
            }

            if (one.SizeX != other.SizeX || one.SizeY != other.SizeY || one.SizeZ != other.SizeZ || one.Name != other.Name)
            {
                Console.WriteLine("!!!!!!!!!!!!!!!!!!!!!");
                Console.WriteLine("!!!!INVALID STATE!!!!");
                Console.WriteLine("!!!!!!!!!!!!!!!!!!!!!");
                throw new Exception("Invalid state!!");
            }
        }

        private ExampleBase.RunResult runExample(ExampleBase example, Execute mode, double sizeFactor)
        {
            example.ExecuteOn = mode;

            System.Threading.Thread.Sleep(100);

            try
            {
                return example.Run(sizeFactor, sizeFactor, sizeFactor, print, rounds, warmup_rounds);
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception.ToString());
                return null;
            }
        }
    }
}
