using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Gpu
{
    public class GpuComputeDevice : ComputeDevice
    {
        public static List<GpuComputeDevice> GpuComputeDevices()
        {
            List<GpuComputeDevice> result = new List<GpuComputeDevice>();

            OpenCLNet.Platform[] platforms = OpenCLNet.OpenCL.GetPlatforms();
            foreach (OpenCLNet.Platform platform in platforms)
            {
                OpenCLNet.Device[] devices = platform.QueryDevices(OpenCLNet.DeviceType.ALL);

                foreach (OpenCLNet.Device device in devices)
                    if (device.DeviceType == OpenCLNet.DeviceType.GPU)
                        result.Add(new GpuComputeDevice(device));
            }

            return result;
        }

        public GpuComputeDevice(OpenCLNet.Device device)
        {
            DeviceType = getDeviceType(device);

            Name = device.Name;
            Manufacturer = device.Vendor;

            DeviceId = device.DeviceID.ToString(); // TODO: get real device id from handle

            ComputeUnits = new List<ComputeUnit>();
            for (int i = 0; i < device.MaxComputeUnits; i++)
                ComputeUnits.Add(new GpuComputeUnit(device));

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
    }
}
