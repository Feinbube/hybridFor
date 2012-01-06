using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class LocalFunctionCall: ExampleBase
    {
        int[] a;
        int[] b;
        int[] c;

        protected override void setup()
        {
            a = new int[sizeX];
            for (int x = 0; x < sizeX; x++)
                a[x] = random.Next(100);

            b = new int[sizeX];
            for (int x = 0; x < sizeX; x++)
                b[x] = random.Next(100);

            c = new int[sizeX];
        }

        protected override void printInput()
        {
            printField(a, sizeX);
            printField(b, sizeX);
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeX, delegate(int i)
            {
                c[i] = theFunction(a[i], b[i]);
            });
        }

        int theFunction(int a, int b)
        {
            return a + b;
        }

        protected override void printResult()
        {
            printField(c, sizeX);
        }

        protected override bool isValid()
        {
            for (int x = 0; x < sizeX; x++)
                if (a[x] + b[x] != c[x])
                    return false;

            return true;
        }
    }
}
