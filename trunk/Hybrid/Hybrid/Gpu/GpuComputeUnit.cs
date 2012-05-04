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

            SharedMemory = getSharedMemory(device);
            Caches = getCaches(device);

        }

        private List<MemoryInfo> getCaches(OpenCLNet.Device device)
        {
            List<MemoryInfo> result = new List<MemoryInfo>();

            if (device.GlobalMemCacheSize > 0)
                result.Add(
                    new MemoryInfo(MemoryInfo.Type.Cache, mapCacheAccess(device.GlobalMemCacheType), device.GlobalMemCacheSize)
                );

            result.Add(
                new MemoryInfo(MemoryInfo.Type.Cache, MemoryInfo.Access.ReadOnly, device.MaxConstantBufferSize)
            );

            return result;
        }

        private MemoryInfo getSharedMemory(OpenCLNet.Device device)
        {
            switch (device.LocalMemType)
            {
                case OpenCLNet.DeviceLocalMemType.GLOBAL:
                    return null;
                case OpenCLNet.DeviceLocalMemType.LOCAL:
                    return new MemoryInfo(MemoryInfo.Type.Shared,device.LocalMemSize);
                default:
                    throw new Exception("LocalMemType " + device.LocalMemType.ToString() + " is unknown.");
            }
        }

        private uint getProcessingElementCount(OpenCLNet.Device device)
        {
            return 8; // TODO: Get Real Values.
        }

        private MemoryInfo.Access mapCacheAccess(OpenCLNet.DeviceMemCacheType deviceMemCacheType)
        {
            if (deviceMemCacheType == OpenCLNet.DeviceMemCacheType.READ_ONLY_CACHE)
                return MemoryInfo.Access.ReadOnly;

            if (deviceMemCacheType == OpenCLNet.DeviceMemCacheType.READ_WRITE_CACHE)
                return MemoryInfo.Access.ReadWrite;

            throw new Exception("OpenCLNet.DeviceMemCacheType " + deviceMemCacheType + " unknown.");
        }
    }
}
