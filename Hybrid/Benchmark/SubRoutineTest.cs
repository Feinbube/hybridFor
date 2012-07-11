using System;
using System.Collections.Generic;
using System.Text;

namespace Hybrid.Benchmark
{
    internal class SubRoutineTest : Hybrid.Examples.ExampleBase
    {
        protected int[] a, b, c;
        private static int special = 1;
        private int d = 2;
        private int[,] e = new int[2, 2] { { 0, 1 }, { 2, 3 } };
        private uint[] r;

        private int AddTwoInts_2ndFromArray(int a, int[] b, int i)
        {
            return AddTwoInts(a, b[i]);
        }

        private int AddTwoInts(int a, int b)
        {
            uint z = Hybrid.MsilToOpenCL.OpenClFunctions.rnd();
            r[Hybrid.MsilToOpenCL.OpenClFunctions.get_global_id(0)] = z;
            return a + b + special + d + e[a % 2, b % 2];
        }

        protected override void setup()
        {
            a = new int[sizeX];
            for (int i = 0; i < sizeX; i++)
                a[i] = random.InstanceNext();

            b = new int[sizeX];
            for (int i = 0; i < sizeX; i++)
                b[i] = random.InstanceNext();

            c = new int[sizeX];
            r = new uint[sizeX];
        }

        protected override void cleanup()
        {
            a = b = c = null;
            r = null;
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeX, delegate(int i)
            {
                int[] a = this.a;
                c[i] = AddTwoInts_2ndFromArray(a[i], b, i);
            });
        }

        protected override void printInput()
        {
            for (int i = 0; i < sizeX; i++)
                Console.Write("{0}+{1} ", a[i], b[i]);
            Console.WriteLine();
        }

        protected override void printResult()
        {
            for (int i = 0; i < sizeX; i++)
                Console.Write("{0} ", c[i]);
            Console.WriteLine();
        }

        protected override bool isValid()
        {
            for (int i = 0; i < sizeX; i++)
            {
                if (a[i] + b[i] + special + d + e[a[i]%2,b[i]%2] != c[i])
                    return false;
            }
            return true;
        }
    }
}