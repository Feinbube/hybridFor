using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Upcrc2010.MatrixMultiplication
{
    public class MatrixMultiplication0 : MatrixMultiplicationBase
    {
        protected override void algorithm() // http://www.codeproject.com/KB/cs/aforge_parallel.aspx
        {
            Parallel.For(ExecuteOn, 0, sizeX, delegate(int i)
            {
                for (int j = 0; j < sizeY; j++)
                {
                    double v = 0;

                    for (int k = 0; k < sizeZ; k++)
                    {
                        v += a[i, k] * b[k, j];
                    }

                    c[i, j] = v;
                }
            });
        }
    }
}
