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
    public class Everywhere : ExampleTestBase
    {
        override protected void testExample(ExampleBase example)
        {
            example.ExecuteOn = Execute.OnEverythingAvailable;

            if (!example.Run(10.0, 10.0, 10.0, false, 5, 2).Valid)
                throw new Exception("Invalid result for " + example.GetType());

            Parallel.ReInitialize();
        }
    }
}
