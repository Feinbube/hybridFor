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

            uint numberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());

            ComputeUnits = new List<ComputeUnit>();
            for (int i = 0; i < numberOfCores; i++)
                ComputeUnits.Add(new CpuComputeUnit(processorInfo));

            GlobalMemory = new MemoryInfo()
            {
                MemoryType = MemoryInfo.Type.Global,
                Size = getGlobalMemorySizeForProcessor(processorInfo)
            };

            Caches = new List<MemoryInfo>{ // where does L2Cache really reside: on-core or on-die?
                new MemoryInfo()
                {
                    MemoryType = MemoryInfo.Type.Cache,
                    Size = uint.Parse(processorInfo["L3CacheSize"].ToString())
                }
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

        override public void ParallelFor(int fromInclusive, int toExclusive, Action<int> action)
        {
            CpuParallelFor(fromInclusive, toExclusive, action);
        }

        public static void CpuParallelFor(int fromInclusive, int toExclusive, Action<int> action)
        {
            if (fromInclusive >= toExclusive)
                return;

            System.Threading.Tasks.Parallel.For(fromInclusive, toExclusive, action);
        }

        override public void ParallelFor(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            CpuParallelFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
        }

        public static void CpuParallelFor(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            if (fromInclusiveX >= toExclusiveX)
                return;

            if (fromInclusiveY >= toExclusiveY)
                return;

            System.Threading.Tasks.Parallel.For(fromInclusiveX, toExclusiveX, delegate(int x)
            {
                System.Threading.Tasks.Parallel.For(fromInclusiveY, toExclusiveY, delegate(int y)
                {
                    action(x, y);
                });
            });
        }
    }
}
