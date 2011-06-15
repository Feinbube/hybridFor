using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Upcrc2010.MatrixMultiplication
{
    public class MatrixMultiplication3 : MatrixMultiplicationBase
    {
        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            double factor = 50.0;
            this.sizeX = (int)(sizeX * factor);
            this.sizeY = (int)(sizeY * factor);
            this.sizeZ = (int)(sizeZ * factor);
        }

        protected override void algorithm() // http://myssa.upcrc.illinois.edu/files/Lab_OpenMP_Assignments/
        {
            int ITILE2 = 32;
            int JTILE2 = 32;

            Parallel.For(ExecuteOn, 0, sizeX, delegate(int ii)
            {
                for (int jj = 0; jj < sizeY; jj += JTILE2)
                {
                    int il = Math.Min((ii * ITILE2) + ITILE2, sizeX);
                    int jl = Math.Min(jj + JTILE2, sizeY);
                    for (int i = (ii * ITILE2); i < il; i++)
                        for (int j = jj; j < jl; j++)
                        {
                            c[i, j] = 0;
                            for (int k = 0; k < sizeZ; k++)
                                c[i, j] += a[i, k] * b[k, j];
                        }
                }
            });
        }
    }
}
