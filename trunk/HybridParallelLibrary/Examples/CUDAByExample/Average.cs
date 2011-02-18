using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using HybridParallelLibrary;

namespace ForGpu.Examples.CudaByExample
{
    public class Average : ExampleBase // Based on CUDA By Example by Jason Sanders and Edward Kandrot
    {
        float[] a;
        float[] b;
        float[] c;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            this.sizeX = (int)(sizeX * 10000000.0);
        }

        protected override void setup()
        {
            a = new float[sizeX];
            for (int i = 0; i < sizeX; i++)
                a[i] = random.Next(0, 1000);

            b = new float[sizeX];
            for (int i = 0; i < sizeX; i++)
                b[i] = random.Next(0, 1000);

            c = new float[sizeX];
        }

        protected override void printInput()
        {
            printField(a, sizeX);
            Console.WriteLine();
            printField(b, sizeX);
        }

        protected override void algorithm()
        {
            Parallel.For(0, sizeX, delegate(int idx)
            {
                int idx1 = (idx + 1) % sizeX;
                int idx2 = (idx + 2) % sizeX;

                float _as = (a[idx] + a[idx1] + a[idx2])/3.0f;
                float bs = (b[idx] + b[idx1] + b[idx2])/3.0f;

                c[idx] = (_as + bs) / 2.0f;
            });
        }

        protected override void printResult()
        {
            printField(c, sizeX);
        }

        protected override bool isValid()
        {
            for (int idx = 0; idx < sizeX; idx++)
            {
                int idx1 = (idx + 1) % sizeX;
                int idx2 = (idx + 2) % sizeX;

                float _as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
                float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;

                if(c[idx] - (_as + bs) / 2.0f > 0.0001f) // TODO Understand where the difference comes from
                    return false;
            }

            return true;
        }
    }
}
