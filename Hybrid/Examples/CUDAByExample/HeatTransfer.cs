using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.CudaByExample
{
    public class HeatTransfer : ExampleBase // Based on CUDA By Example by Jason Sanders and Edward Kandrot
    {
        float[,] input;
        float[,] bitmap;

        protected override void setup()
        {
            this.sizeZ = this.sizeX / 10;

            input = new float[sizeX, sizeY];
            bitmap = new float[sizeX, sizeY];

            float scale = 5.0f;

            for (int x = 0; x < sizeX; x++)
                for (int y = 0; y < sizeY; y++)
                {
                    input[x, y] = 0.0f;

                    if (x > sizeX / 2.0f - sizeX / scale && x < sizeX / 2.0f + sizeX / scale &&
                        y > sizeY / 2.0f - sizeY / scale && y < sizeY / 2.0f + sizeY / scale)
                        input[x, y] = 255.0f;
                }
        }

        protected override void printInput()
        {
            paintField(input);
        }

        protected override void algorithm()
        {
            float speed = 0.25f;

            float[,] temp;

            for (int i = 0; i < sizeZ; i++)
            {
                Parallel.For(ExecuteOn, 0, sizeX, 0, sizeY, delegate(int x, int y)
                {
                    int left = Math.Max(0, x - 1);
                    int right = Math.Min(sizeX - 1, x + 1);
                    int top = Math.Max(0, y - 1);
                    int bottom = Math.Min(sizeY - 1, y + 1);

                    bitmap[x, y] = input[x, y] + speed * (input[x, top] + input[x, bottom] + input[left, y] + input[right, y] - input[x, y] * 4.0f);
                });

                temp = bitmap;
                bitmap = input;
                input = temp;
            }
        }

        protected override void printResult()
        {
            paintField(bitmap);
        }

        protected override bool isValid()
        {
            // TODO implement check
            return true;
        }
    }
}
