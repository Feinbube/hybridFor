using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Upcrc2010
{
    public class Convolution : ExampleBase // http://myssa.upcrc.illinois.edu/files/Lab_OpenMP_Assignments/
    {
        double[,] startImage;
        double[,] outImage;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            double factor = 2000.0;
            this.sizeX = (int)(sizeX * factor);
            this.sizeY = (int)(sizeY * factor);
        }

        protected override void setup()
        {
            double EDGE = 5.0;

            startImage = new double[sizeX, sizeY];
            outImage = new double[sizeX, sizeY];

            // create random, directed weight matrix
            for (int i = 0; i < sizeX; i++)
                for (int j = 0; j < sizeY; j++)
                    startImage[i, j] = random.NextDouble() * 10.0;

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
            Parallel.For(1, sizeX - 1, 1, sizeY - 1, delegate(int i, int j)
            {
                outImage[i, j] = (1.0 / 5.0) * (startImage[i, j] + startImage[i, j + 1] + startImage[i, j - 1] + startImage[i + 1, j] + startImage[i - 1, j]);
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
                    if (outImage[i, j] != (1.0 / 5.0) * (startImage[i, j] + startImage[i, j + 1] + startImage[i, j - 1] + startImage[i + 1, j] + startImage[i - 1, j]))
                        return false;
                }
            }

            return true;
        }
    }
}
