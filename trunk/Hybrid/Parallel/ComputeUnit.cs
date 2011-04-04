using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid
{
    public class ComputeUnit
    {
        public bool AtomicsSupported;
        public bool DoublePrecisionFloatingPointSupported;

        public MemoryInfo SharedMemory;
        public MemoryInfo GlobalMemoryCache;

        public List<ProcessingElement> ProcessingElements;

        public ComputeUnit(ManagementObject processorInfo)
        {
            uint NumberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());
            uint NumberOfLogicalProcessors = uint.Parse(processorInfo["NumberOfLogicalProcessors"].ToString());

            uint processingElementCount = NumberOfLogicalProcessors / NumberOfCores;

            ProcessingElements = new List<ProcessingElement>();
            for (int i = 0; i < processingElementCount; i++)
                ProcessingElements.Add(new ProcessingElement(processorInfo));

            SharedMemory = null;

            GlobalMemoryCache = new MemoryInfo()
            {
                Size = uint.Parse(processorInfo["L3CacheSize"].ToString()),
                Type = MemoryInfo.Types.ReadWriteCache
            };
        }

        public ComputeUnit(OpenCLNet.Device device)
        {
            AtomicsSupported = device.Extensions.Contains("atomics");
            DoublePrecisionFloatingPointSupported = device.Extensions.Contains("fp64");

            ProcessingElements = new List<ProcessingElement>();
            for (int i = 0; i < getProcessingElementCount(device); i++)
                ProcessingElements.Add(new ProcessingElement(device));

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

        public double PredictPerformance(AlgorithmCharacteristics algorithmCharacteristics)
        {
            double result = 0.0;

            foreach (ProcessingElement processingElement in ProcessingElements)
                result += processingElement.ClockSpeed;

            if (algorithmCharacteristics.UsesDoublePrecisionFloatingPoint && !DoublePrecisionFloatingPointSupported)
                result /= 8; // TODO: check heuristics

            if (algorithmCharacteristics.UsesAtomics && !AtomicsSupported)
                result = 0;

            // TODO: consider memory as well

            return result;
        }
    }
}
