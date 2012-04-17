using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class ConvolutionNPP : ExampleBase // http://myssa.upcrc.illinois.edu/files/Lab_OpenMP_Assignments/
    {
        byte[,] startImage;
        byte[,] outImage;

        protected override void setup()
        {
            byte EDGE = 128;

            startImage = new byte[sizeX, sizeY];
            outImage = new byte[sizeX, sizeY];

            // create random, directed weight matrix
            for (int i = 0; i < sizeX; i++)
                for (int j = 0; j < sizeY; j++)
                    startImage[i, j] = (byte) Random.Next();

            //Set up edge values within border elements
            for (int k = 0; k < sizeX; k++)
            {
                startImage[k, 0] = outImage[k, 0] = EDGE;
                startImage[k, sizeY - 1] = outImage[k, sizeY - 1] = EDGE;
            }

            for (int k = 0; k < sizeY; k++)
            {
                startImage[0, k] = outImage[0, k] = EDGE;
                startImage[sizeX - 1, k] = outImage[sizeX - 1, k] = EDGE;
            }
        }

        protected override void printInput()
        {
            printField(startImage, sizeX, sizeY);
        }

        protected override void algorithm()
        {
            // assumes "padding" to avoid messy border cases   
            Parallel.For(ExecuteOn, 1, sizeX - 1, 1, sizeY - 1, delegate(int i, int j)
            {
                int sum = startImage[i - 1, j - 1] + startImage[i + 1, j - 1] + startImage[i - 1, j + 1] + startImage[i + 1, j + 1] + startImage[i, j] + startImage[i, j + 1] + startImage[i, j - 1] + startImage[i + 1, j] + startImage[i - 1, j];
                outImage[i, j] = (byte)(sum/9);
            });
        }

        protected override void printResult()
        {
            printField(outImage, sizeX, sizeY);
        }

        protected override bool isValid()
        {
            for (int i = 1; i < sizeX - 1; i++)
            {
                for (int j = 1; j < sizeY - 1; j++)
                {
                    int sum = startImage[i - 1, j - 1] + startImage[i + 1, j - 1] + startImage[i - 1, j + 1] + startImage[i + 1, j + 1] + startImage[i, j] + startImage[i, j + 1] + startImage[i, j - 1] + startImage[i + 1, j] + startImage[i - 1, j];
                    if (outImage[i, j] != (byte) (sum/9))
                        return false;
                }
            }

            return true;
        }

        protected override void cleanup()
        {
            startImage = null;
            outImage = null;
        }
    }
}
