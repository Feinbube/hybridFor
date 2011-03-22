using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class ILBB
    {
        public int? stackCountOnEntry;
        public int? stackCountOnExit;
        public int offset;
        public List<CilInstruction> list = new List<CilInstruction>();
        public CilInstruction FinalTransfer;

        public ILBB FallThroughTarget;
    }
}
