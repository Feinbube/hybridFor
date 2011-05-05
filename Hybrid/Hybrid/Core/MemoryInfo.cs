using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class MemoryInfo
    {
        public enum Types { ReadOnlyCache, ReadWriteCache, Global, Shared, Private }
        public Types Type;

        public ulong Size;
    }
}
