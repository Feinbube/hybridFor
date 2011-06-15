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
    public class MultiGpuExampleTests
    {
        [TestMethod]
        public void MultiGpu_StaticFunctionCall()
        {
            testExample(new StaticFunctionCall());
        }

        [TestMethod]
        public void MultiGpu_LocalFunctionCall()
        {
            testExample(new LocalFunctionCall());
        }

        [TestMethod]
        public void MultiGpu_MatrixMultiplicationTest()
        {
            testExample(new MatrixMultiplication0());
            testExample(new MatrixMultiplication1());
            testExample(new MatrixMultiplication2());
            testExample(new MatrixMultiplication3());
            testExample(new MatrixMultiplication4());
            testExample(new MatrixMultiplication5());
        }

        [TestMethod]
        public void MultiGpu_AverageTest()
        {
            testExample(new Average());
        }

        [TestMethod]
        public void MultiGpu_DotProductTest()
        {
            testExample(new DotProduct());
        }

        [TestMethod]
        public void MultiGpu_HeatTransferTest()
        {
            testExample(new HeatTransfer());
        }

        [TestMethod]
        public void MultiGpu_HistogramTest()
        {
            testExample(new Histogram());
        }

        [TestMethod,Ignore]
        public void MultiGpu_JuliaSetTest()
        {
            testExample(new JuliaSet());
        }

        [TestMethod,Ignore]
        public void MultiGpu_RayTracingTest()
        {
            testExample(new RayTracing());
        }

        [TestMethod]
        public void MultiGpu_RippleTest()
        {
            testExample(new Ripple());
        }

        [TestMethod]
        public void MultiGpu_SummingVectorsTest()
        {
            testExample(new SummingVectors());
        }

        [TestMethod]
        public void MultiGpu_ConvolutionTest()
        {
            testExample(new Convolution());
        }

        [TestMethod]
        public void MultiGpu_MatrixVectorMultiplicationTest()
        {
            testExample(new MatrixVectorMultiplication());
        }

        [TestMethod]
        public void MultiGpu_MinimumSpanningTreeTest()
        {
            testExample(new MinimumSpanningTree());
        }

        [TestMethod]
        public void MultiGpu_PrefixScanTest()
        {
            testExample(new PrefixScan());
        }

        [TestMethod]
        public void MultiGpu_GameOfLifeTest()
        {
            testExample(new GameOfLife());
        }

        [TestMethod]
        public void MultiGpu_WatorTest()
        {
            testExample(new Wator());
        }

        [TestMethod]
        public void MultiGpu_QuickSortTest()
        {
            testExample(new QuickSort());
        }

        private void testExample(ExampleBase example)
        {
            example.ExecuteOn = Execute.OnAllGpus;

            if(example.Run(0.1, 0.1, 0.1, false, 5, 2, null) < 0 )
                throw new Exception("Invalid result for " + example.GetType());

            Parallel.ReInitialize();
        }
    }
}
