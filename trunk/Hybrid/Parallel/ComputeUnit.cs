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

        public List<ProcessingElement> ProcessingElements;

        public ComputeUnit(ManagementObject processorInfo)
        {
            uint NumberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());
            uint NumberOfLogicalProcessors = uint.Parse(processorInfo["NumberOfLogicalProcessors"].ToString());

            uint processingElementCount = NumberOfLogicalProcessors / NumberOfCores;

            for (int i = 0; i < processingElementCount; i++)
                ProcessingElements.Add(new ProcessingElement(processorInfo));

            //uint L2CacheSize = uint.Parse(processorInfo["L2CacheSize"].ToString());
            //uint L3CacheSize = uint.Parse(processorInfo["L3CacheSize"].ToString());
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
                    Size = device.LocalMemSize,

                    CacheType = MemoryInfo.CacheTypes.None,
                    CacheSize = 0,
                    CacheLineSize = 0
                };
        }

        private uint getProcessingElementCount(OpenCLNet.Device device)
        {
            return 8; // TODO: Get Real Values.
        }
    }
}
