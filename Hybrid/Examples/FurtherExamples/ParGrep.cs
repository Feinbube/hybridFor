using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class ParGrep: ExampleBase
    {
        byte needle;
        byte[] haystack;
        int[] positions;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            double factor = 200000.0;
            this.sizeX = (int)(sizeX * factor);
            this.sizeY = -1; // unused
            this.sizeZ = -1; // unused
        }

        protected override void setup()
        {
            haystack = new byte[sizeX];
            positions = new int[sizeX];

            for (int x = 0; x < sizeX; x++)
                haystack[x] = (byte)random.Next(256);

            needle = (byte)random.Next(256);
        }

        protected override void printInput()
        {
            printField(haystack, sizeX);
        }

        protected override void algorithm()
        {
            /* Serial Implementation:
             * 
             * for (int x = 0; x < sizeX; x++)
             *    if (haystack[x] == needle)
             *        positions[k++] = x;
             *        
             */

            int k = 0;
            
            Parallel.For(ExecuteOn, 0, sizeX, delegate(int x)
            {
                if (haystack[x] == needle)
                {
                    positions[k] = x;
                    
                    // TODO postpone incrementing of k.
                    // TODO create local results first and insert them afterwards.
                    Atomic.Add(ref k, 1);
                }
            });
        }

        protected override void printResult()
        {
            printField(positions, sizeX);
        }

        protected override bool isValid()
        {
            for (int x = 0; x < sizeX; x++)
            {
                if (positions[x] == 0)
                    break; // reached the end of the results

                if (haystack[positions[x]] != needle)
                    return false;
            }

            return true;
        }
    }
}
