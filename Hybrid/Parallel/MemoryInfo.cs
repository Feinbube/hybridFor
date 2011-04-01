using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class MemoryInfo
    {
        public enum CacheTypes { None, ReadOnly, ReadWrite }
        public CacheTypes CacheType;

        public ulong CacheSize;
        public ulong CacheLineSize;

        public ulong Size;
    }
}
