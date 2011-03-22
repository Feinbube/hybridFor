using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Hybrid;
using Hybrid.Examples;
using Hybrid.Examples.Upcrc2010.MatrixMultiplication;
using Hybrid.Examples.CudaByExample;
using Hybrid.Examples.Upcrc2010;

namespace Hybrid.Testsuite
{
    [TestClass]
    public class Examples
    {
        [TestMethod]
        public void MatrixMultiplicationTest()
        {
            testExample(new MatrixMultiplication0());
            testExample(new MatrixMultiplication1());
            testExample(new MatrixMultiplication2());
            testExample(new MatrixMultiplication3());
            testExample(new MatrixMultiplication4());
            testExample(new MatrixMultiplication5());
        }

        [TestMethod]
        public void AverageTest()
        {
            testExample(new Average());
        }

        [TestMethod]
        public void DotProductTest()
        {
            testExample(new DotProduct());
        }

        [TestMethod]
        public void HeatTransferTest()
        {
            testExample(new HeatTransfer());
        }

        [TestMethod,Ignore]
        public void JuliaSetTest()
        {
            testExample(new JuliaSet());
        }

        [TestMethod,Ignore]
        public void RayTracingTest()
        {
            testExample(new RayTracing());
        }

        [TestMethod]
        public void RippleTest()
        {
            testExample(new Ripple());
        }

        [TestMethod]
        public void SummingVectorsTest()
        {
            testExample(new SummingVectors());
        }

        [TestMethod]
        public void ConvolutionTest()
        {
            testExample(new Convolution());
        }

        [TestMethod]
        public void MatrixVectorMultiplicationTest()
        {
            testExample(new MatrixVectorMultiplication());
        }

        [TestMethod]
        public void MinimumSpanningTreeTest()
        {
            testExample(new MinimumSpanningTree());
        }

        [TestMethod]
        public void PrefixScanTest()
        {
            testExample(new PrefixScan());
        }

        [TestMethod]
        public void QuickSortTest()
        {
            testExample(new QuickSort());
        }

        private void testExample(ExampleBase example)
        {
            runExample(example, ExecutionMode.Serial);
            runExample(example, ExecutionMode.Gpu);
            runExample(example, ExecutionMode.TaskParallel);
            runExample(example, ExecutionMode.TaskParallel2D);
            runExample(example, ExecutionMode.Gpu2D);
            
            Parallel.ReInitialize();
        }

        private void runExample(ExampleBase forGpuExample, ExecutionMode mode)
        {
            Parallel.Mode = mode;
            Atomic.Mode = mode;

            if(forGpuExample.Run(0.1, 0.1, 0.1, false, 5, 2, null) < 0 )
                throw new Exception("Invalid result for " + forGpuExample.GetType());

            System.Threading.Thread.Sleep(20);
        }
    }
}
