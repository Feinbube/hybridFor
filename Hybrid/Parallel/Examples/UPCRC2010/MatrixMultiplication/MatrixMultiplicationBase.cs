using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Upcrc2010.MatrixMultiplication
{
    public abstract class MatrixMultiplicationBase : ExampleBase
    {
        protected double[,] a;
        protected double[,] b;

        protected double[,] c;

        protected override void setup()
        {
            a = new double[sizeX, sizeZ];
            
            for (int i = 0; i < sizeX; i++)
                for (int j = 0; j < sizeZ; j++)
                    a[i, j] = random.NextDouble();

            b = new double[sizeZ, sizeY];

            for (int i = 0; i < sizeZ; i++)
                for (int j = 0; j < sizeY; j++)
                    b[i, j] = random.NextDouble();

            c = new double[sizeX, sizeY];
        }

        protected override void printInput()
        {
            printField(a, sizeX, sizeZ);
            printField(b, sizeZ, sizeY);
        }

        protected override void printResult()
        {
            printField(c, sizeX, sizeY);
        }

        protected override bool isValid()
        {
            // http://www.codeproject.com/KB/cs/aforge_parallel.aspx

            for (int i = 0; i < sizeX; i++)
            {
                for (int j = 0; j < sizeY; j++)
                {
                    double v = 0;

                    for (int k = 0; k < sizeZ; k++)
                        v += a[i, k] * b[k, j];

                    if (Math.Abs(c[i, j] - v) > 0.00000000005f)
                        return false;
                }
            }

            return true;
        }
    }
}
