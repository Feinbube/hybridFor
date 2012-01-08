using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class Lists: ExampleBase
    {
        List<int> a;
        List<int> b;

        protected override void setup()
        {
            a = new List<int>();
            for (int x = 0; x < sizeX; x++)
                a.Add(random.Next(100));

            b = new List<int>();
            for (int x = 0; x < sizeX; x++)
                b.Add(0);
        }

        protected override void printInput()
        {
            printField(a.ToArray(), sizeX);
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeX, delegate(int i)
            {
                b[i] = a[i] * a[i];
            });
        }

        protected override void printResult()
        {
            printField(a.ToArray(), sizeX);
        }

        protected override bool isValid()
        {
            for (int x = 0; x < sizeX; x++)
                if (a[x]*a[x] != b[x])
                    return false;

            return true;
        }

        protected override void cleanup()
        {
            a.Clear();
            a = null;

            b.Clear();
            b = null;
        }
    }
}
