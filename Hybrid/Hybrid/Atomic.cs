using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Hybrid.MsilToOpenCL;

namespace Hybrid
{
    public class Atomic
    {
        [OpenClAlias("atom_add")]
        public static int Add(ref int location1, int value)
        {
            // due to the attribute above, this will never be replaced for Gpus
            return System.Threading.Interlocked.Add(ref location1, value);
        }
    }
}
