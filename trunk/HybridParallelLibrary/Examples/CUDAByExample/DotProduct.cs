using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using HybridParallelLibrary;

namespace Examples.CudaByExample
{
    public class DotProduct : ExampleBase // Based on CUDA By Example by Jason Sanders and Edward Kandrot
    {
        float[] a;
        float[] b;
        float result;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            double factor = 3000.0;
            this.sizeX = (int)(sizeX * factor);
            this.sizeY = (int)(sizeY * factor);
        }

        protected override void setup()
        {
            a = new float[sizeX * sizeY];
            b = new float[sizeX * sizeY];

            for (int i = 0; i < sizeX * sizeY; i++)
            {
                a[i] = i;
                b[i] = i * 2;
            }
        }

        protected override void printInput()
        {
            printField(a, sizeX * sizeY);
            Console.WriteLine();
            printField(b, sizeX * sizeY);
        }

        protected override void algorithm()
        {
            float[] temp = new float[sizeX];

            // TODO the book also provides a Multi-GPU version
            // How can we support that also? Explicit? Automatic?
            Parallel.For(0, sizeX, delegate(int x)
            {
                temp[x] = 0.0f;
                int tid = x;

                for (int y = 0; y < sizeY; y++)
                {
                    temp[x] += a[tid] * b[tid];
                    tid += sizeX;
                }
            });

            result = 0.0f;

            // apply reduction here
            for (int x = 0; x < sizeX; x++)
                result += temp[x];
        }

        protected override void printResult()
        {
            Console.WriteLine(result);
        }

        protected override bool isValid()
        {
            float result = 0.0f;

            for (int i = 0; i < sizeX * sizeY; i++)
                result += a[i] * b[i];

            if (this.result / result < 0.99999f)
                return false;
            else
                return true;
        }
    }
}
