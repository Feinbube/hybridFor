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
    public class Gpu : ExampleTestBase
    {
        override protected void testExample(ExampleBase example)
        {
            example.ExecuteOn = Execute.OnSingleGpu;

            if(example.Run(0.1, 0.1, 0.1, false, 5, 2, null) < 0 )
                throw new Exception("Invalid result for " + example.GetType());
            
            Parallel.ReInitialize();
        }
    }
}
