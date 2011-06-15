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
    abstract public class ExampleTestBase
    {
        [TestMethod]
        public void Lists()
        {
            testExample(new Lists());
        }

        [TestMethod]
        public void StaticFunctionCall()
        {
            testExample(new StaticFunctionCall());
        }

        [TestMethod]
        public void LocalFunctionCall()
        {
            testExample(new LocalFunctionCall());
        }

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

        [TestMethod]
        public void HistogramTest()
        {
            testExample(new Histogram());
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
        public void GameOfLifeTest()
        {
            testExample(new GameOfLife());
        }

        [TestMethod]
        public void WatorTest()
        {
            testExample(new Wator());
        }

        [TestMethod]
        public void QuickSortTest()
        {
            testExample(new QuickSort());
        }

        protected abstract void testExample(ExampleBase example);
    }
}
