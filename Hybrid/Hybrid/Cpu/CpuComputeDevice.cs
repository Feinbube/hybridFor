using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid.Cpu
{
    public class CpuComputeDevice : ComputeDevice
    {
        public static List<CpuComputeDevice> CpuComputeDevices()
        {
            List<CpuComputeDevice> result = new List<CpuComputeDevice>();

            ManagementClass processorInfos = new ManagementClass("Win32_Processor");
            foreach (ManagementObject processorInfo in processorInfos.GetInstances())
                result.Add(new CpuComputeDevice(processorInfo));

            return result;
        }

        public CpuComputeDevice(ManagementObject processorInfo)
        {
            DeviceType = DeviceTypes.Cpu;

            Name = processorInfo["Name"].ToString();
            Manufacturer = processorInfo["Manufacturer"].ToString();
            DeviceId = processorInfo["DeviceID"].ToString();

            uint NumberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());

            ComputeUnits = new List<ComputeUnit>();
            for (int i = 0; i < NumberOfCores; i++)
                ComputeUnits.Add(new CpuComputeUnit(processorInfo));

            GlobalMemory = new MemoryInfo()
            {
                Type = MemoryInfo.Types.Global,
                Size = getGlobalMemorySizeForProcessor(processorInfo)
            };
        }

        private ulong getGlobalMemorySizeForProcessor(ManagementObject processorInfo)
        {
            ulong overallCapacity = 0; // TODO: NUMA-aware

            ManagementClass physicalMemoryInfos = new ManagementClass("Win32_PhysicalMemory");
            foreach (ManagementObject physicalMemoryInfo in physicalMemoryInfos.GetInstances())
                overallCapacity += ulong.Parse(physicalMemoryInfo["Capacity"].ToString());

            ManagementClass processorInfos = new ManagementClass("Win32_Processor");
            int processorCount = processorInfos.GetInstances().Count;

            return overallCapacity / (ulong)processorCount;
        }
    }
}
