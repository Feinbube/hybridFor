using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Functionality 
{
    public class ForLoop : ExampleBase
    {
        int[] a;
        int [] result;
        protected override void setup()
        {
            if (sizeX > Int32.MaxValue || sizeX < 0)
                sizeX = Int32.MaxValue;
            a = new int[sizeX];
            result = new int[sizeX];
            for (int x = 0; x < sizeX; x++)
            {
                a[x] = x;
                result[x] = 0;
            }
        }

        protected override void printInput()
        {
            printField(a, sizeX);
        }

        protected override void algorithm()
        {
            for (int x = 0; x < sizeX; x++)
            {
                result[x] = 0;
            }   
            Parallel.For(ExecuteOn,0,sizeX,delegate(int x){
                for(int i = 0; i < 100; i++)
                {
                    result[x] += a[x];
                }
            });
        }

        protected override void printResult()
        {
            Console.WriteLine(result);
        }


        protected override void cleanup()
        {
            a = null;
            result = null;
        }

        protected override bool isValid()
        {
            for (int x = 0; x < sizeX; x++)
            {
                if ( result[x] != 100 * a[x])
                {
                    return false;
                }
            }
            return true;
        }

    }
}
