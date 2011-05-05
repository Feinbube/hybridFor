using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid.Cpu
{
    public class CpuProcessingElement:ProcessingElement
    {
        public uint ClockSpeed;

        public MemoryInfo PrivateMemory;
        public MemoryInfo Cache;

        public CpuProcessingElement(ManagementObject processorInfo)
        {
            uint CurrentClockSpeed = uint.Parse(processorInfo["CurrentClockSpeed"].ToString());
            uint MaxClockSpeed = uint.Parse(processorInfo["MaxClockSpeed"].ToString());

            ClockSpeed = MaxClockSpeed;

            PrivateMemory = null; // TODO: get real values

            Cache = new MemoryInfo()
            {
                Type = MemoryInfo.Types.ReadWriteCache,
                Size = uint.Parse(processorInfo["L2CacheSize"].ToString())
            };
        }
    }
}
