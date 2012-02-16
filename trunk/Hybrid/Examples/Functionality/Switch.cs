using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    // Simple test for switch statement
    // Just fill an input array with random values,
    // Iterate across it and "identify" 0s and 1s using a switch statement.
    public class Switch: ExampleBase
    {
        int[] input;
        int[] output;

        protected override void setup()
        {
            if (sizeX > 16777216)
                sizeX = 16777216;

            input = new int[sizeX];
            output = new int[sizeX];

            for (int i = 0; i < sizeX; i++)
                input[i] = Random.Next(10);
        }

        protected override void printInput()
        {
            printField(input, sizeX);
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeX, delegate(int x)
            {
                switch (input[x])
                {
                    case 0:
                        output[x] = -1;
                        break;
                    case 1:
                        output[x] = +1;
                        break;
                    default:
                        output[x] = 0;
                        break;
                }
            });
        }

        protected override void printResult()
        {
            printField(output, sizeX);
        }

        protected override bool isValid()
        {
            for (int x = 0; x < sizeX; x++)
            {
                switch (input[x])
                {
                    case 0:
                        if (output[x] != -1)
                            return false;
                        break;
                    case 1:
                        if (output[x] != +1)
                            return false;
                        break;
                    default:
                        if (output[x] != 0)
                            return false;
                        break;
                }
            }
            return true;
        }

        protected override void cleanup()
        {
            input = null;
            output = null;
        }
    }
}
