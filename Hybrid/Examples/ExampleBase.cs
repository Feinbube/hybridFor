using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace Hybrid.Examples
{
    public abstract class ExampleBase
    {
        protected Random random = new Random();

        protected int sizeX = 0;
        protected int sizeY = 0;
        protected int sizeZ = 0;

        public double Run(double size, bool print, int rounds)
        {
            return Run(size, size, size, print, rounds, 0, null);
        }

        public double Run(double sizeX, double sizeY, double sizeZ, bool print, int rounds, int warmupRounds, TextWriter tw)
        {
            Console.Write("Running " + this.GetType().Name);

            scaleAndSetSizes(sizeX, sizeY, sizeZ);

            Console.Write("(" + this.sizeX + " / " + this.sizeY + " / " + this.sizeZ + ")...");
            
            setup();

            if (print)
            {
                Console.WriteLine();
                Console.WriteLine("Input:");
                printInput();
            }

            for (int warmupRound = 0; warmupRound < warmupRounds; warmupRound++)
                algorithm();

            System.Diagnostics.Stopwatch Watch = new System.Diagnostics.Stopwatch();
            Watch.Start();
            for (int round = 0; round < rounds; round++)
                algorithm();
            Watch.Stop();

            for (int warmupRound = 0; warmupRound < warmupRounds; warmupRound++)
                algorithm();

            if (print)
            {
                Console.WriteLine();
                Console.WriteLine("Result:");
                printResult();
            }

            bool valid = checkResult(print);

            Console.WriteLine("Done in " + Watch.Elapsed.TotalSeconds + "s. " + (valid ? "SUCCESS" : "<!!! FAILED !!!>"));

            if (tw != null)
            {
                tw.WriteLine(this.GetType().Name + ";" + this.sizeX + ";" + this.sizeY + ";" + this.sizeZ + ";" + Parallel.Mode + "_" + Atomic.Mode + ";" + Watch.Elapsed.TotalMilliseconds + ";" + valid);
                tw.Flush();
            }

            if (!valid)
                return -1;
            else
                return Watch.Elapsed.TotalSeconds;
        }

        protected abstract void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ);
        protected abstract void setup();
        protected abstract void printInput();
        protected abstract void algorithm();
        protected abstract void printResult();

        protected bool checkResult(bool throwOnError)
        {
            if (!isValid())
            {
                if (throwOnError)
                {
                    throw new Exception("Calculated Result is not valid.");  // this line gets annoying after some time...
                }
                return false;
            }

            return true;
        }

        protected abstract bool isValid();

        protected string doubleToString(double value)
        {
            return String.Format("{0:0.00}", value).Substring(0, 4);
        }

        protected void swap(ref byte[,] a, ref byte[,] b)
        {
            byte[,] tmp = a;
            a = b;
            b = tmp;
        }

        protected void printField(double[,] field, int sizeX, int sizeY)
        {
            for (int i = -1; i <= sizeX; i++)
            {
                for (int j = -1; j <= sizeY; j++)
                {
                    if (j == -1 || j == sizeY || i == -1 || i == sizeX)
                        Console.Write("**** ");
                    else
                        Console.Write(doubleToString(field[i, j]) + " ");
                }
                Console.WriteLine();
            }
        }

        protected void printField(byte[,] fields, int sizeX, int sizeY, Action<int,int> printAction)
        {
            for (int y = 0; y < sizeY; y++)
            {
                for (int x = 0; x < sizeX; x++)
                    printAction.Invoke(x, y);
                Console.WriteLine();
            }
        }

        protected void printField(float[] field, int sizeX)
        {
            for (int i = 0; i < Math.Min(sizeX, 80); i++)
                Console.Write(doubleToString(field[i]) + " ");

            if (sizeX > 80)
                Console.Write("...");

            Console.WriteLine();
        }

        protected void printField(byte[] field, int sizeX)
        {
            int[] intArray = new int[sizeX];

            for (int i = 0; i < sizeX; i++)
                intArray[i] = field[i];

            printField(intArray, sizeX);
        }

        protected void printField(int[] field, int sizeX)
        {
            for (int i = 0; i < Math.Min(sizeX, 80); i++)
                Console.Write(field[i] + " ");

            if (sizeX > 80)
                Console.Write("...");

            Console.WriteLine();
        }

        protected void paintField(float[,] bitmap)
        {
            for (int x = 0; x < sizeX; x++)
                for (int y = 0; y < sizeY; y++)
                {
                    paintValue(bitmap[x, y], 0.0f, 32.0f, ' ');
                    paintValue(bitmap[x, y], 32.0f, 64.0f, '.');
                    paintValue(bitmap[x, y], 64.0f, 96.0f, '°');
                    paintValue(bitmap[x, y], 96.0f, 128.0f, ':');
                    paintValue(bitmap[x, y], 128.0f, 160.0f, '+');
                    paintValue(bitmap[x, y], 160.0f, 192.0f, '*');
                    paintValue(bitmap[x, y], 192.0f, 224.0f, '#');
                    paintValue(bitmap[x, y], 224.0f, 256.0f, '8');
                }
        }

        protected void paintValue(double value, double fromInclusive, double toExclusive, char output)
        {
            if (value >= fromInclusive && value < toExclusive)
                Console.Write(output);
        }
    }
}
