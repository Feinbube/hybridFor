using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class MemoryInfo
    {
        public enum Access { ReadOnly, WriteOnly, ReadWrite }
        public enum Type { Global, Shared, Private, Cache }
        public enum PreferredAccessPattern { Linear, Areal } // to reflect constant memory and texture memory

        public Type MemoryType = Type.Global;
        public Access MemoryAccess = Access.ReadWrite;
        public PreferredAccessPattern MemoryPreferredAccessPattern = PreferredAccessPattern.Linear;

        public ulong Size;
        public ulong LineSize = 0;
    }
}
