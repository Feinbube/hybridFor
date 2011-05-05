using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Gpu
{
    public class GpuProcessingElement : ProcessingElement
    {
        public uint ClockSpeed;

        public MemoryInfo PrivateMemory;
        public MemoryInfo Cache;

        public GpuProcessingElement(OpenCLNet.Device device)
        {
            ClockSpeed = device.MaxClockFrequency;

            PrivateMemory = null; // TODO: get real values
        }
    }
}
