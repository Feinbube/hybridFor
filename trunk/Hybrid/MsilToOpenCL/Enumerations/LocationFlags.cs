using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    [Flags]
    public enum LocationFlags
    {
        Read = 1,
        IndirectRead = 2,
        Write = 4,
        IndirectWrite = 8,
        ForcePointer = 0x10,
        PointerGlobal = 0x20,
    }
}
