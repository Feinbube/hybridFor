using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.CudaByExample
{
    public class Ripple : ExampleBase // Based on CUDA By Example by Jason Sanders and Edward Kandrot
    {
        float[,] bitmap;

        protected override void setup()
        {
            bitmap = new float[sizeX, sizeY];
        }

        protected override void printInput()
        {
            Console.WriteLine("SizeX:" + sizeX + " SizeY:" + sizeY);
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeX, 0, sizeY, delegate(int x, int y)
            {
                float fx = x - sizeX / 2.0f;
                float fy = y - sizeY / 2.0f;

                float d = (float)Math.Sqrt(fx * fx + fy * fy);

                int ticks = 1;
                float grey = (float)(128.0f + 127.0f * Math.Cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

                bitmap[x, y] = grey;
            });
        }

        protected override void printResult()
        {
            paintField(bitmap);
        }

        protected override bool isValid()
        {
            for (int x = 0; x < sizeX; x++)
                for (int y = 0; y < sizeY; y++)
                {
                    float fx = x - sizeX / 2.0f;
                    float fy = y - sizeY / 2.0f;

                    float d = (float)Math.Sqrt(fx * fx + fy * fy);

                    int ticks = 1;
                    float grey = (float)(128.0f + 127.0f * Math.Cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

					if (Math.Abs(bitmap[x, y] - grey) > 0.00005f)
                        return false;
                }

            return true;
        }

        protected override void cleanup()
        {
            bitmap = null;
        }
    }
}
