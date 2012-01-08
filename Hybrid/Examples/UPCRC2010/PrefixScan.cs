using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Upcrc2010
{
    public class PrefixScan : ExampleBase // http://myssa.upcrc.illinois.edu/files/Lab_OpenMP_Assignments/
    {
        float[] startData;
        float[] IscanData;

        protected override void setup()
        {
            if (sizeX > 10000000 || sizeX < 0)
                sizeX = 10000000;

            startData = new float[sizeX];
            IscanData = new float[sizeX];

            for (int i = 0; i < sizeX; i++)
                startData[i] = (float)(random.NextDouble() * 10.0);
        }

        protected override void printInput()
        {
            Console.WriteLine("Starting array:");
            printField(startData, sizeX);
        }

        protected override void algorithm()
        {
            float[] temp = new float[sizeX];

            int increment = 1;

            float[] p1 = temp;
            float[] p2 = IscanData;
            float[] ptmp;

            for (long i = 0; i < sizeX; ++i)
            {
                temp[i] = startData[i];
            }

            IscanData[0] = startData[0];

            while (increment < sizeX)
            {

                Parallel.For(ExecuteOn, 1, increment, delegate(int i)
                {
                    p2[i] = p1[i];
                });

                Parallel.For(ExecuteOn, increment, sizeX, delegate(int i)
                {
                    p2[i] = p1[i] + p1[i - increment];
                });

                // increment
                increment = increment << 1;

                // switch arrays
                ptmp = p1;
                p1 = p2;
                p2 = ptmp;
            }
        }

        protected override void printResult()
        {
            printField(IscanData, sizeX);
        }

        protected override bool isValid()
        {
            for (int i = 0; i < sizeX/2 /* FIX ME */; i++)
                if (IscanData[i + 1]/IscanData[i] < 0.99999f)
                    return false;

            return true;
        }
    }
}
