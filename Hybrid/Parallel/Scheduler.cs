using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid
{
    public class Scheduler
    {
        List<ExecutionDevice> executionDevices = new List<ExecutionDevice>();

        public Scheduler()
        {
            findOpenCLDevices();

            if (noCpuInDeviceList())
                findProcessors();
        }

        private bool noCpuInDeviceList()
        {
            foreach (ExecutionDevice executionDevice in executionDevices)
                if (executionDevice.DeviceType == ExecutionDevice.DeviceTypes.Cpu)
                    return true;

            return false;
        }

        private void findProcessors()
        {
            ManagementClass processorInfos = new ManagementClass("Win32_Processor");
            foreach (ManagementObject processorInfo in processorInfos.GetInstances())
            {
                uint NumberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());
                uint NumberOfLogicalProcessors = uint.Parse(processorInfo["NumberOfLogicalProcessors"].ToString());

                string Name = processorInfo["Name"].ToString();
                string Manufacturer = processorInfo["Manufacturer"].ToString();

                uint CurrentClockSpeed = uint.Parse(processorInfo["CurrentClockSpeed"].ToString());
                uint MaxClockSpeed = uint.Parse(processorInfo["MaxClockSpeed"].ToString());

                uint L2CacheSize = uint.Parse(processorInfo["L2CacheSize"].ToString());
                uint L3CacheSize = uint.Parse(processorInfo["L3CacheSize"].ToString());

                executionDevices.Add(new ExecutionDevice()
                {
                    DeviceType = ExecutionDevice.DeviceTypes.Cpu,

                    Name = Name,
                    Manufacturer = Manufacturer,

                    NumberOfCores = NumberOfCores,
                    NumberOfLogicalProcessors = NumberOfLogicalProcessors,

                    CurrentClockSpeed = CurrentClockSpeed,
                    MaxClockSpeed = MaxClockSpeed,

                    MemoryList = new List<Memory>
                    {
                        new Memory(){
                            MemoryType = Memory.MemoryTypes.L2Cache,
                            Size = (ulong)L2CacheSize
                        },
                        new Memory(){
                            MemoryType = Memory.MemoryTypes.L3Cache,
                            Size = (ulong)L3CacheSize
                        }
                    }
                });
            }
        }

        private void findOpenCLDevices()
        {
            OpenCLNet.Platform[] platforms = OpenCLNet.OpenCL.GetPlatforms();
            foreach (OpenCLNet.Platform platform in platforms)
            {
                OpenCLNet.Device[] devices = platform.QueryDevices(OpenCLNet.DeviceType.ALL);
                foreach (OpenCLNet.Device device in devices)
                {
                    executionDevices.Add(new ExecutionDevice()
                    {
                        DeviceType = getDeviceType(device),

                        Name = device.Name,
                        Manufacturer = device.Vendor,

                        NumberOfCores = device.MaxComputeUnits,

                        MaxClockSpeed = device.MaxClockFrequency,

                        MemoryList = new List<Memory>
                        {
                            new Memory(){
                                MemoryType = Memory.MemoryTypes.Global,
                                Size = device.GlobalMemSize,
                                CacheSize= device.GlobalMemCacheSize,
                                CacheLineSize = device.GlobalMemCacheLineSize
                            }

                            throw new Exception("HERE");
                        }
                    });
                }
            }
        }

        private static ExecutionDevice.DeviceTypes getDeviceType(OpenCLNet.Device device)
        {
            ExecutionDevice.DeviceTypes deviceType = ExecutionDevice.DeviceTypes.Unknown;

            if (device.DeviceType == OpenCLNet.DeviceType.CPU)
                deviceType = ExecutionDevice.DeviceTypes.Cpu;

            if (device.DeviceType == OpenCLNet.DeviceType.GPU)
                deviceType = ExecutionDevice.DeviceTypes.Gpu;

            if (device.DeviceType == OpenCLNet.DeviceType.ACCELERATOR)
                deviceType = ExecutionDevice.DeviceTypes.Accelerator;

            return deviceType;
        }

        public static void ExecuteAutomatic(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            for (int x = fromInclusiveX; x < toExclusiveX; x++)
                for (int y = fromInclusiveY; y < toExclusiveY; y++)
                    action(x, y);
        }
    }
}
