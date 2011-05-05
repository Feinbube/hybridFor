using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid
{
    public class Platform
    {
        public List<ComputeDevice> ComputeDevices = new List<ComputeDevice>();

        public Platform()
        {
            findOpenCLDevices();

            if (noCpuInDeviceList())
                findProcessors();
        }

        private void findOpenCLDevices()
        {
            OpenCLNet.Platform[] platforms = OpenCLNet.OpenCL.GetPlatforms();
            foreach (OpenCLNet.Platform platform in platforms)
            {
                OpenCLNet.Device[] devices = platform.QueryDevices(OpenCLNet.DeviceType.ALL);

                foreach (OpenCLNet.Device device in devices)
                    if(device.DeviceType != OpenCLNet.DeviceType.CPU) // We don't like the way AMD and Intel map CPUs to OpenCL
                        ComputeDevices.Add(new ComputeDevice(device));
            }
        }

        private bool noCpuInDeviceList()
        {
            foreach (ComputeDevice computeDevice in ComputeDevices)
                if (computeDevice.DeviceType == ComputeDevice.DeviceTypes.Cpu)
                    return false;

            return true;
        }

        private void findProcessors()
        {
            ManagementClass processorInfos = new ManagementClass("Win32_Processor");
            foreach (ManagementObject processorInfo in processorInfos.GetInstances())
                ComputeDevices.Add(new ComputeDevice(processorInfo));
        }

        public double PredictPerformance(AlgorithmCharacteristics algorithmCharacteristics)
        {
            double result = 0.0;

            foreach (ComputeDevice computeDevice in ComputeDevices)
                result += computeDevice.PredictPerformance(algorithmCharacteristics);

            return result;
        }
    }
}
