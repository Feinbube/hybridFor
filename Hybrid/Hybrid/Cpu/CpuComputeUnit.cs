using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid.Cpu
{
    public class CpuComputeUnit : ComputeUnit
    {
        public CpuComputeUnit(ManagementObject processorInfo)
        {
            uint numberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());
            uint numberOfLogicalProcessors = uint.Parse(processorInfo["NumberOfLogicalProcessors"].ToString());

            uint processingElementCount = numberOfLogicalProcessors / numberOfCores;

            ProcessingElements = new List<ProcessingElement>();
            for (int i = 0; i < processingElementCount; i++)
                ProcessingElements.Add(new CpuProcessingElement(processorInfo));

            SharedMemory = null;

            Caches = new List<MemoryInfo>{ // what about L1Cache?
                new MemoryInfo() // where does L2Cache really reside: on-core or on-die?
                {
                    Type = MemoryInfo.Types.ReadWriteCache,
                    Size = uint.Parse(processorInfo["L2CacheSize"].ToString())
                }
            };
        }
    }
}
