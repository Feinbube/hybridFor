using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class MemoryInfo
    {
        /*
        device.GlobalMemCacheLineSize;
            device.GlobalMemCacheSize;
            device.GlobalMemCacheType == OpenCLNet.DeviceMemCacheType.READ_ONLY_CACHE
                */

        public enum Types { ReadOnlyCache, ReadWriteCache, Global, Shared, Private }
        public Types Type;

        public ulong Size;

    }
}
