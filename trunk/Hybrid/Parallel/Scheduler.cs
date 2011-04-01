using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class Scheduler
    {
        Platform platform = new Platform();

        public static void ExecuteAutomatic(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            for (int x = fromInclusiveX; x < toExclusiveX; x++)
                for (int y = fromInclusiveY; y < toExclusiveY; y++)
                    action(x, y);
        }
    }
}