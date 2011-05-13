using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid.Cpu
{
    public class CpuProcessingElement:ProcessingElement
    {
        public CpuProcessingElement(ManagementObject processorInfo)
        {
            uint CurrentClockSpeed = uint.Parse(processorInfo["CurrentClockSpeed"].ToString());
            uint MaxClockSpeed = uint.Parse(processorInfo["MaxClockSpeed"].ToString());

            ClockSpeed = MaxClockSpeed;
        }
    }
}
