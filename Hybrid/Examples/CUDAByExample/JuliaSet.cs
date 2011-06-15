﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.CudaByExample
{
    public class JuliaSet : ExampleBase // Based on CUDA By Example by Jason Sanders and Edward Kandrot
    {
        struct Complex
        {
            float r;
            float i;

            public Complex(float r, float i)
            {
                this.r = r;
                this.i = i;
            }

            public float magnitude2()
            {
                return r * r + i * i;
            }

            public static Complex operator *(Complex c1, Complex c2)
            {
                return new Complex(c1.r * c2.r - c1.i * c2.i, c1.i * c2.r + c1.r * c2.i);
            }

            public static Complex operator +(Complex c1, Complex c2)
            {
                return new Complex(c1.r + c2.r, c1.i + c2.i);
            }
        }

        int[,] bitmap;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            double factor = 3000.0;
            this.sizeX = (int)(sizeX * factor);
            this.sizeY = (int)(sizeY * factor);
        }

        protected override void setup()
        {
            bitmap = new int[sizeX, sizeY];
        }

        protected override void printInput()
        {
            Console.WriteLine("SizeX:" + sizeX + " SizeY:" + sizeY);
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeY, 0, sizeX, delegate(int y, int x)
            {
                bitmap[x, y] = julia(x, y);
            });

			Parallel.For(ExecuteOn, 0, sizeY, 0, sizeX, delegate(int y, int x) {
				const float scale = 1.5f;

				float jx = scale * (float)(sizeX / 2.0f - x) / (sizeX / 2.0f);
				float jy = scale * (float)(sizeY / 2.0f - y) / (sizeY / 2.0f);

				float cr = -0.8f, ci = 0.156f;
				float ar = jx, ai = jy;

				int v = 1;
				for (int i = 0; i < 200; i++) {
					float nr = ar * ar - ai * ai, ni = ai * ar + ar * ai;
					ar = nr + cr; ai = ni + ci;

					if (ar * ar + ai * ar > 1000) {
						v = 0;
						break;
					}
				}

				bitmap[x, y] = v;
			});
        }

        private int julia(int x, int y)
        {
            const float scale = 1.5f;

            float jx = scale * (float)(sizeX / 2.0f - x) / (sizeX / 2.0f);
            float jy = scale * (float)(sizeY / 2.0f - y) / (sizeY / 2.0f);

            Complex c = new Complex(-0.8f, 0.156f);
            Complex a = new Complex(jx, jy);

            for (int i = 0; i < 200; i++)
            {
                a = a * a + c;
                if (a.magnitude2() > 1000)
                    return 0;
            }

            return 1;
        }

        protected override void printResult()
        {
            for (int y = 0; y < sizeY; y++)
            {
                for (int x = 0; x < sizeX; x++)
                    if (bitmap[x, y] == 0)
                        Console.Write(" ");
                    else
                        Console.Write("*");

                Console.WriteLine();
            }
        }

        protected override bool isValid()
        {
            for (int x = 0; x < sizeX; x++)
                for (int y = 0; y < sizeY; y++)
                    if (bitmap[x, y] != julia(x, y))
                        return false;

            return true;
        }
    }
}
