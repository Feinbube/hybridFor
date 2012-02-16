using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Functionality
{
    public class RandomExample : ExampleBase
    {
        int[] a;
        protected override void setup()
        {
            if (sizeX > Int32.MaxValue || sizeX < 0)
                sizeX = Int32.MaxValue;
            a = new int[sizeX];
            for (int x = 0; x < sizeX; x++)
            {
                a[x] = 0;
            }
        }

        protected override void printInput()
        {
            printField(a, sizeX);
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeX, delegate(int x)
            {
                a[x] = Random.Next();
            });
        }

        protected override void printResult()
        {
            printField(a, sizeX);
        }

        protected override void cleanup()
        {
            a = null;
        }

        protected override bool isValid()
        {
            //check wether all values are the same
            bool check = false;
            for (int i = 1; i < sizeX; i++)
            {
                if (a[i - 1] != a[i])
                {
                    check = true;
                    break;
                }
            }
            return check;
        }

    }
}
