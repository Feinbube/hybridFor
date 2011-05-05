using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class Scheduler
    {
        public static Platform Platform = new Platform();

        public static void ExecuteAutomatic(int fromInclusive, int toExclusive, Action<int> action)
        {
            for (int i = fromInclusive; i < toExclusive; i++)
                action(i);
        }

        public static void ExecuteAutomatic(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            for (int x = fromInclusiveX; x < toExclusiveX; x++)
                for (int y = fromInclusiveY; y < toExclusiveY; y++)
                    action(x, y);
        }
    }
}