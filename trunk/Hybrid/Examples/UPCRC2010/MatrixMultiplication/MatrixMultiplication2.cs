using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Upcrc2010.MatrixMultiplication
{
    public class MatrixMultiplication2 : MatrixMultiplicationBase
    {
        protected override void algorithm() // http://myssa.upcrc.illinois.edu/files/Lab_OpenMP_Assignments/
        {
            Parallel.For(ExecuteOn, 0, sizeX, 0, sizeY, delegate(int i, int j)
            {
                    c[i, j] = 0;
                    for (int k = 0; k < sizeZ; k++)
                        c[i, j] += a[i, k] * b[k, j];
            });
        }
    }
}
