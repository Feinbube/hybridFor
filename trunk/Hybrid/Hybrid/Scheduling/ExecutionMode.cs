using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public enum ExecutionMode{ Serial, TaskParallel, TaskParallel2D, Gpu, Gpu2D, MultiGpu, Automatic }
    public enum ExecuteOn { Cpu, Gpu, All } // supposed to replace the one above
}
