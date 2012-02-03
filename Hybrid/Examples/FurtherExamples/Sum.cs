using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.FurtherExamples
{
    class Sum: ExampleBase
    {
        int[] a;
        int result;

        protected override void setup()
        {
            if (sizeX > Int32.MaxValue || sizeX < 0)
                sizeX = Int32.MaxValue;

            a = new int[sizeX];
            for (int i = 0; i < sizeX; i++)
            {
               a[i] = i;
            }
            result = 0;
        }

        protected override void printInput()
        {
            printField(a, sizeX);
            Console.WriteLine();
        }

        protected override void algorithm() 
        {
            for (int i = 0; i < a.Length; i++)
            {
                result += a[i];
            }
        }

        protected override bool isValid()
        {
            int gaussSum = (int) sizeX * (sizeX - 1) / 2;
            if (result == gaussSum)
            {
                return true;
            }
            return false;
        }

        protected override void printResult()
        {
            Console.WriteLine(result);
        }

        protected override void cleanup()
        {
            a = null;
        }
    }
}
