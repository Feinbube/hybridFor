using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using HybridParallelLibrary;

namespace ForGpu.Examples.Upcrc2010.MatrixMultiplication
{
    public class MatrixMultiplication1 : MatrixMultiplicationBase
    {
        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            double factor = 450.0;
            this.sizeX = (int)(sizeX * factor);
            this.sizeY = (int)(sizeY * factor);
            this.sizeZ = (int)(sizeZ * factor);
        }

        protected override void algorithm() // http://www.codeproject.com/KB/cs/aforge_parallel.aspx
        {
            Parallel.For(0, sizeX, 0, sizeY, delegate(int i, int j)
            {
                double v = 0;

                for (int k = 0; k < sizeZ; k++)
                    v += a[i, k] * b[k, j];

                c[i, j] = v;
            });
        }
    }
}
