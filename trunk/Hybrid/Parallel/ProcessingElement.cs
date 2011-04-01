using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid
{
    public class ProcessingElement
    {
        public uint ClockSpeed;

        public MemoryInfo PrivateMemory;

        public ProcessingElement(ManagementObject processorInfo)
        {
            uint CurrentClockSpeed = uint.Parse(processorInfo["CurrentClockSpeed"].ToString());
            uint MaxClockSpeed = uint.Parse(processorInfo["MaxClockSpeed"].ToString());
        }

        public ProcessingElement(OpenCLNet.Device device)
        {
            ClockSpeed = device.MaxClockFrequency;
            PrivateMemory = null; // TODO: get real values
        }
    }
}
