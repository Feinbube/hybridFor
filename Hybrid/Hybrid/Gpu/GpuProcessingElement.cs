using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Gpu
{
    public class GpuProcessingElement : ProcessingElement
    {
        public GpuProcessingElement(OpenCLNet.Device device)
        {
            ClockSpeed = device.MaxClockFrequency;
        }
    }
}
