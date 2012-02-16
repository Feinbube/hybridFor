using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Functionality
{
    public class AtomicExample : ExampleBase
    {
        int[] a;
        int result;
        protected override void setup()
        {
            if (sizeX > Int32.MaxValue || sizeX < 0)
                sizeX = Int32.MaxValue;
            a = new int[sizeX];
            for (int x = 0; x < sizeX; x++)
            {
                a[x] = 1;
            }
            result = 0;
        }

        protected override void printInput()
        {
            printField(a, sizeX);
        }

        protected override void algorithm()
        {
            result = 0;
            Parallel.For(ExecuteOn,0,sizeX,delegate(int x){
                Atomic.Add(ref result, a[x]);
            });
        }

        protected override void printResult()
        {
            Console.WriteLine(result);
        }


        protected override void cleanup()
        {
            a = null;
        }

        protected override bool isValid()
        {
            if (result == sizeX)
            {
                return true;
            }
            return false;
        }

    }
}
