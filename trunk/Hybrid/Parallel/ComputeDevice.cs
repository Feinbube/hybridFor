using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;
using System.Runtime.InteropServices;

namespace Hybrid
{
    public class ComputeDevice
    {
        public enum DeviceTypes { Cpu, Gpu, Accelerator, Unknown }
        public DeviceTypes DeviceType;

        public string Name;
        public string Manufacturer;
        public string DeviceId;

        public MemoryInfo GlobalMemory;

        public List<ComputeUnit> ComputeUnits;

        public ComputeDevice(ManagementObject processorInfo)
        {
            DeviceType = DeviceTypes.Cpu;

            Name = processorInfo["Name"].ToString();
            Manufacturer = processorInfo["Manufacturer"].ToString();
            DeviceId = processorInfo["DeviceID"].ToString();

            uint NumberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());

            ComputeUnits = new List<ComputeUnit>();
            for (int i = 0; i < NumberOfCores; i++)
                ComputeUnits.Add(new ComputeUnit(processorInfo));

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

        public ComputeDevice(OpenCLNet.Device device)
        {
            DeviceType = getDeviceType(device);

            Name = device.Name;
            Manufacturer = device.Vendor;

            DeviceId = device.DeviceID.ToString(); // TODO: get real device id from handle

            ComputeUnits = new List<ComputeUnit>();
            for (int i = 0; i < device.MaxComputeUnits; i++)
                ComputeUnits.Add(new ComputeUnit(device));

            GlobalMemory = new MemoryInfo()
            {
                Type = MemoryInfo.Types.Global,
                Size = device.GlobalMemSize
            };
        }

        private ComputeDevice.DeviceTypes getDeviceType(OpenCLNet.Device device)
        {
            ComputeDevice.DeviceTypes deviceType = ComputeDevice.DeviceTypes.Unknown;

            if (device.DeviceType == OpenCLNet.DeviceType.CPU)
                deviceType = ComputeDevice.DeviceTypes.Cpu;

            if (device.DeviceType == OpenCLNet.DeviceType.GPU)
                deviceType = ComputeDevice.DeviceTypes.Gpu;

            if (device.DeviceType == OpenCLNet.DeviceType.ACCELERATOR)
                deviceType = ComputeDevice.DeviceTypes.Accelerator;

            return deviceType;
        }

        public double PredictPerformance(AlgorithmCharacteristics algorithmCharacteristics)
        {
            double result = 0.0;

            foreach (ComputeUnit computeUnit in ComputeUnits)
                result += computeUnit.PredictPerformance(algorithmCharacteristics);

            // TODO: consider DeviceTypes
            // TODO: consider Memory

            return result;
        }
    }
}