using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid.Cpu
{
    public class CpuComputeUnit:ComputeUnit
    {
        public CpuComputeUnit(ManagementObject processorInfo)
        {
            uint NumberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());
            uint NumberOfLogicalProcessors = uint.Parse(processorInfo["NumberOfLogicalProcessors"].ToString());

            uint processingElementCount = NumberOfLogicalProcessors / NumberOfCores;

            ProcessingElements = new List<ProcessingElement>();
            for (int i = 0; i < processingElementCount; i++)
                ProcessingElements.Add(new CpuProcessingElement(processorInfo));

            SharedMemory = null;

            GlobalMemoryCache = new MemoryInfo()
            {
                Size = uint.Parse(processorInfo["L3CacheSize"].ToString()),
                Type = MemoryInfo.Types.ReadWriteCache
            };
        }
    }
}
