using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace HybridParallelLibrary.Gpu
{
    public class Parallel
    {
        public static void For(int fromInclusive, int toExclusive, Action<int> action)
        {
            throw new NotImplementedException();
        }

        public static void For(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            throw new NotImplementedException();
        }

        public static void Invoke(params Action[] actions)
        {
            throw new NotImplementedException();
        }
    }
}
