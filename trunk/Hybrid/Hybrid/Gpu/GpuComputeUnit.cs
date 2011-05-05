using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid.Gpu
{
    public class GpuComputeUnit : ComputeUnit
    {
        public GpuComputeUnit(OpenCLNet.Device device)
        {
            AtomicsSupported = device.Extensions.Contains("atomics");
            DoublePrecisionFloatingPointSupported = device.Extensions.Contains("fp64");

            ProcessingElements = new List<ProcessingElement>();
            for (int i = 0; i < getProcessingElementCount(device); i++)
                ProcessingElements.Add(new GpuProcessingElement(device));

            if (device.LocalMemType == OpenCLNet.DeviceLocalMemType.GLOBAL)
                SharedMemory = null;

            if (device.LocalMemType == OpenCLNet.DeviceLocalMemType.LOCAL)
                SharedMemory = new MemoryInfo()
                {
                    Type = MemoryInfo.Types.Shared,
                    Size = device.LocalMemSize
                };

            if (device.GlobalMemCacheSize == 0)
                GlobalMemoryCache = null;

            if (device.GlobalMemCacheSize > 0)
                GlobalMemoryCache = new MemoryInfo()
                {
                    Type = mapCacheType(device.GlobalMemCacheType),
                    Size = device.GlobalMemCacheSize
                };
        }

        private uint getProcessingElementCount(OpenCLNet.Device device)
        {
            return 8; // TODO: Get Real Values.
        }

        private MemoryInfo.Types mapCacheType(OpenCLNet.DeviceMemCacheType deviceMemCacheType)
        {
            if (deviceMemCacheType == OpenCLNet.DeviceMemCacheType.READ_ONLY_CACHE)
                return MemoryInfo.Types.ReadOnlyCache;

            if (deviceMemCacheType == OpenCLNet.DeviceMemCacheType.READ_WRITE_CACHE)
                return MemoryInfo.Types.ReadWriteCache;

            throw new Exception("OpenCLNet.DeviceMemCacheType " + deviceMemCacheType + " unknown.");
        }
    }
}
