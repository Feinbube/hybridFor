using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid
{
    public class Platform
    {
        List<ComputeDevice> computeDevices = new List<ComputeDevice>();

        public Platform()
        {
            findOpenCLDevices();

            if (noCpuInDeviceList()) // TODO: check if AMDs implementation creates aquivalent ComputeDevices
                findProcessors();
        }

        private void findOpenCLDevices()
        {
            OpenCLNet.Platform[] platforms = OpenCLNet.OpenCL.GetPlatforms();
            foreach (OpenCLNet.Platform platform in platforms)
            {
                OpenCLNet.Device[] devices = platform.QueryDevices(OpenCLNet.DeviceType.ALL);

                foreach (OpenCLNet.Device device in devices)
                    if(device.DeviceType != OpenCLNet.DeviceType.CPU) // We don't like the way AMD maps CPUs to OpenCL
                        computeDevices.Add(new ComputeDevice(device));
            }
        }

        private bool noCpuInDeviceList()
        {
            foreach (ComputeDevice computeDevice in computeDevices)
                if (computeDevice.DeviceType == ComputeDevice.DeviceTypes.Cpu)
                    return false;

            return true;
        }

        private void findProcessors()
        {
            ManagementClass processorInfos = new ManagementClass("Win32_Processor");
            foreach (ManagementObject processorInfo in processorInfos.GetInstances())
                computeDevices.Add(new ComputeDevice(processorInfo));
        }
    }
}
