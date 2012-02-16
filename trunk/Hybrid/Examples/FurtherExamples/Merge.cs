using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class Merge: ExampleBase
    {
        int[] input1, input2;
        int[] output;

        protected override void setup()
        {
            if (sizeX > 500000) sizeX = 500000;
            input1 = new int[sizeX];
            input2 = new int[sizeX];
            output = new int[2 * sizeX];

            for (int x = 0; x < sizeX; x++)
            {
                input1[x] = Random.Next();
                input2[x] = Random.Next();
            }
            Array.Sort(input1);
            Array.Sort(input2);
        }

        protected override void printInput()
        {
            printField(input1, sizeX);
            printField(input2, sizeX);
        }

        // TODO find a parallel implementation
        protected override void algorithm()
        {
            int i, j, k;
            i = j = k = 0;

            while (i < sizeX && j < sizeX)
            {
                if (input1[i] < input2[j])
                    output[k++] = input1[i++];
                else if (input1[i] > input2[j])
                    output[k++] = input2[j++];
                else // (input1[i] == input2[j])
                {
                    output[k++] = input1[i++];
                    //j++;
                }
            }

            // add missing elements to output
            while (i < sizeX)
                output[k++] = input1[i++];

            while(j < sizeX)
                output[k++] = input2[j++];
        }

        protected override void printResult()
        {
            printField(output, 2 * sizeX);
        }

        protected override bool isValid()
        {
            // output needs to contain all values of input{1,2}
            // this validation takes forever, because it is in O(n^2)
            for (int x = 0; x < sizeX; x++)
                if (!output.Contains(input1[x]) || !output.Contains(input2[x]))
                    return false;

            // output needs to be sorted 
            for (int x = 0; x < 2 * sizeX - 1; x++)
                if (output[x] > output[x + 1])
                    return false;

            return true;
        }

        protected override void cleanup()
        {
            input1 = null;
            input2 = null;
            output = null;
        }
    }
}
