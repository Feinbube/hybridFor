using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using HybridParallelLibrary;

namespace Examples.CudaByExample
{
    public class SummingVectors : ExampleBase // Based on CUDA By Example by Jason Sanders and Edward Kandrot
    {
        int[] a;
        int[] b;
        int[] c;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            this.sizeX = (int)(sizeX * 5000000.0);
        }

        protected override void setup()
        {
            a = new int[sizeX];
            b = new int[sizeX];
            c = new int[sizeX];

            for (int i = 0; i < sizeX; i++)
            {
                a[i] = -i;
                b[i] = i * i;
            }
        }

        protected override void printInput()
        {
            printField(a, sizeX);
            Console.WriteLine();
            printField(b, sizeX);
        }

        protected override void algorithm()
        {
            Parallel.For(0, sizeX, delegate(int tid)
            {
                c[tid] = a[tid] + b[tid];
            });
        }

        protected override void printResult()
        {
            printField(c, sizeX);

        }

        protected override bool isValid()
        {
            for (int i = 0; i < sizeX; i++)
                if (c[i] != a[i] + b[i])
                    return false;

            return true;
        }
    }
}
