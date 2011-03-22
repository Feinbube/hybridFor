using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class Atomic
    {
        public static ExecutionMode Mode = ExecutionMode.TaskParallel;

        [Ever.OpenClAlias("atom_add")]
        public static int Add(ref int location1, int value)
        {
            if (Mode == ExecutionMode.TaskParallel || Mode == ExecutionMode.TaskParallel2D || Mode == ExecutionMode.Serial)
                return System.Threading.Interlocked.Add(ref location1, value);

            if (Mode == ExecutionMode.Gpu || Mode == ExecutionMode.Gpu2D)
                return Gpu.Atomic.Add(ref location1, value);

            throw new Exception("ParallelMode " + Mode + " is not known.");
        }
    }
}
