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
        //Functionality Tests

        [TestMethod]
        public void Lists()
        {
            testExample(new Lists());
        }

        [TestMethod]
        public void LocalFunctionCall()
        {
            testExample(new LocalFunctionCall());
        }

        [TestMethod]
        public void StaticFunctionCall()
        {
            testExample(new StaticFunctionCall());
        }

        [TestMethod]
        public void Switch()
        {
            testExample(new Switch());
        }

        //CUDA Book Examples
        
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


        //UPCRC 2010

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

        // Further Examples

        [TestMethod]
        public void SudokuValidatorTest()
        {
            testExample(new SudokuValidator());
        }

        [TestMethod]
        public void SudokuValidator2DTest()
        {
            testExample(new SudokuValidator2D());
        }

        [TestMethod]
        public void SudokuValidatorInvalidColumnTest()
        {
            testExample(new SudokuValidatorInvalidColumn());
        }

        [TestMethod]
        public void SudokuValidatorInvalidNumbersTest()
        {
            testExample(new SudokuValidatorInvalidNumbers());
        }

        [TestMethod]
        public void SudokuValidatorInvalidRowTest()
        {
            testExample(new SudokuValidatorInvalidRow());
        }

        [TestMethod]
        public void SudokuValidatorInvalidSubfieldTest()
        {
            testExample(new SudokuValidatorInvalidSubfield());
        }

        [TestMethod]
        public void Crypt3Test()
        {
            testExample(new Crypt3());
        }
        
        [TestMethod]
        public void GameOfLifeTest()
        {
            testExample(new GameOfLife());
        }

        [TestMethod]
        public void MergeTest()
        {
            testExample(new Merge());
        }

        [TestMethod]
        public void ParGrepTest()
        {
            testExample(new ParGrep());
        }

        [TestMethod]
        public void WatorTest()
        {
            testExample(new Wator());
        }

        protected abstract void testExample(ExampleBase example);
    }
}
