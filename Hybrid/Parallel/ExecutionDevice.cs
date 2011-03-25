using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class ExecutionDevice
    {
        public enum DeviceTypes { Cpu, Gpu, Accelerator, Unknown }

        public DeviceTypes DeviceType;

        public List<Memory> MemoryList = null;

        public string Name;
        public string Manufacturer;

        public uint NumberOfCores;
        public uint NumberOfLogicalProcessors;

        public uint CurrentClockSpeed;
        public uint MaxClockSpeed;
    }
}
