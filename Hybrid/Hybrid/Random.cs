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

        public int InstanceNext()
        {
            return Next();
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
            //uint result = (uint)staticRandom.Next();
            return (uint) GetRandomNumber(Int32.MaxValue);
        }


        // The following method is adapted from 
        // http://blog.codeeffects.com/Article/Generate-Random-Numbers-And-Strings-C-Sharp

        public static int GetRandomNumber(int maxNumber)
        {
            if (maxNumber < 1)
                throw new System.Exception("The maxNumber value should be greater than 1");
            byte[] b = new byte[4];
            new System.Security.Cryptography.RNGCryptoServiceProvider().GetBytes(b);
            int seed = (b[0] & 0x7f) << 24 | b[1] << 16 | b[2] << 8 | b[3];
            System.Random r = new System.Random(seed);
            return r.Next(1, maxNumber);
        }
    }
}
