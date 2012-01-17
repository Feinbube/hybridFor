using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Hybrid.MsilToOpenCL;

namespace Hybrid
{
    public class Random
    {
        static System.Random staticRandom = new System.Random();

        public static void Seed(int seed)
        {
            staticRandom = new System.Random(seed);
        }

        public static int Next()
        {
            return (int)NextUInt();
        }

        public static int Next(int maxValue)
        {
            return (int)(NextDouble() * maxValue);
        }

        public static int Next(int minValue, int maxValue)
        {
            return (int)(minValue + (maxValue-minValue)*NextDouble());
        }

        public static void NextBytes(byte[] array)
        {
            for (int i = 0; i < array.Length; i++)
                array[i] = (byte)(NextDouble() * 255);
        }

        public static double NextDouble()
        {
            return (double)Random.NextUInt()/((double)(uint.MaxValue)+1.0);
        }

        [OpenClAlias("MWC64X")]
        public static uint NextUInt()
        {
            return (uint)staticRandom.NextDouble();
        }
    }
}
