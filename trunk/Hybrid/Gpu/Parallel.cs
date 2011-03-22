using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Gpu
{
    public class Parallel
    {
        public static void For(int fromInclusive, int toExclusive, Action<int> action)
        {
            Hybrid.MsilToOpenCL.Parallel.ForGpu(fromInclusive, toExclusive, action);
        }

        public static void For(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            Hybrid.MsilToOpenCL.Parallel.ForGpu(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
        }

        public static void Invoke(params Action[] actions)
        {
            // TODO: Uncomment me
            // throw new NotImplementedException();

            System.Threading.Tasks.Parallel.Invoke(actions);
        }

        public static void ReInitialize()
        {
            Hybrid.MsilToOpenCL.Parallel.PurgeCaches();
        }
    }
}
