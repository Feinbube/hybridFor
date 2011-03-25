using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class Memory
    {
        public enum MemoryTypes { L2Cache, L3Cache, Register, Private, Local, Global }

        public MemoryTypes MemoryType;

        public ulong Size;
        public ulong CacheSize;
        public ulong CacheLineSize;
    }
}
