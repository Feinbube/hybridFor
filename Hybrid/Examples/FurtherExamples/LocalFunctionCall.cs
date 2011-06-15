using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class LocalFunctionCall: ExampleBase
    {
        List<int> a;
        List<int> b;
        List<int> c;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            this.sizeX = (int)(sizeX * 100);
        }

        protected override void setup()
        {
            init(a);
            init(b);

            c = new List<int>();
        }

        private void init(List<int> list)
        {
            list = new List<int>();
            for (int x = 0; x < sizeX; x++)
                list.Add(random.Next(100));
        }

        protected override void printInput()
        {
            printField(a.ToArray(), sizeX);
            printField(b.ToArray(), sizeX);
        }

        protected override void algorithm()
        {
            Parallel.For(0, sizeX, delegate(int i)
            {
                c.Add(theFunction(a[i], b[i]));
            });
        }

        int theFunction(int a, int b)
        {
            return a + b;
        }

        protected override void printResult()
        {
            printField(c.ToArray(), sizeX);
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
