using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.CudaByExample
{
    public class RayTracing : ExampleBase // Based on CUDA By Example by Jason Sanders and Edward Kandrot
    {
        struct Sphere
        {
            public float r, g, b;
            public float radius;
            public float x, y, z;

            public float hit(float ox, float oy, ref float n)
            {
                float dx = ox - x;
                float dy = oy - y;

                if (dx * dx + dy * dy < radius * radius)
                {
                    float dz = (float)Math.Sqrt(radius * radius - dx * dx - dy * dy);
                    n = (float)(dz / Math.Sqrt(radius * radius));
                    return dz + z;
                }

                return float.MinValue;
            }

            public void print()
            {
                //Console.Write("[(" + r + "," + g + "," + b + ")/" + radius + "/(" + x + "," + y + "," + z + ")]");
                Console.Write("[(" + r + ")/" + radius + "/(" + x + "," + y + "," + z + ")]");
            }
        }

        Sphere[] sphere;
        float[,] bitmap;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            this.sizeX = (int)(sizeX * 500.0);
            this.sizeY = (int)(sizeY * 500.0);
            this.sizeZ = (int)(sizeZ * 500.0);
        }

        protected override void setup()
        {
            // for nice rendering of the examples
            //this.sizeX = Math.Max(this.sizeX, 40);
            //this.sizeY = Math.Max(this.sizeY, 80);
            //this.sizeZ = Math.Max(this.sizeZ, 100);

            bitmap = new float[sizeX, sizeY];

            sphere = new Sphere[sizeZ];

            for (int i = 0; i < sizeZ; i++)
            {
                sphere[i] = new Sphere();

                sphere[i].r = (float)random.NextDouble();
                sphere[i].g = (float)random.NextDouble();
                sphere[i].b = (float)random.NextDouble();

                sphere[i].x = (float)(random.NextDouble() * 100.0f - 50.0f);
                sphere[i].y = (float)(random.NextDouble() * 100.0f - 50.0f);
                sphere[i].z = (float)(random.NextDouble() * 100.0f - 50.0f);

                sphere[i].radius = (float)(random.NextDouble() * 10.0f + 2.0f);
            }
        }

        protected override void printInput()
        {
            for (int i = 0; i < Math.Min(sizeZ, 10); i++)
                sphere[i].print();

            Console.WriteLine();
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeX, 0, sizeY, delegate(int x, int y)
            {
                float ox = x - sizeX / 2.0f;
                float oy = y - sizeY / 2.0f;

                float r = 0.0f;
                float g = 0.0f;
                float b = 0.0f;

                float maxz = float.MinValue;

                for (int i = 0; i < sizeZ; i++)
                {
                    float n = 0.0f;
                    float t = sphere[i].hit(ox, oy, ref n);

                    if (t > maxz)
                    {
                        float fscale = n;
                        r = sphere[i].r * fscale;
                        g = sphere[i].g * fscale;
                        b = sphere[i].b * fscale;
                    }
                }

                bitmap[x, y] = r * 256 *256 * 256 + g * 256 + b;
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
                    float ox = x - sizeX / 2.0f;
                    float oy = y - sizeY / 2.0f;

                    float r = 0.0f;
                    float g = 0.0f;
                    float b = 0.0f;

                    float maxz = float.MinValue;

                    for (int i = 0; i < sizeZ; i++)
                    {
                        float n = 0.0f;
                        float t = sphere[i].hit(ox, oy, ref n);

                        if (t > maxz)
                        {
                            float fscale = n;
                            r = sphere[i].r * fscale;
                            g = sphere[i].g * fscale;
                            b = sphere[i].b * fscale;
                        }
                    }

                    if (Math.Abs(bitmap[x, y] - (r * 256 *256 * 256 + g * 256 + b)) > 1)
                        return false;
                }

            return true;
        }
    }
}
