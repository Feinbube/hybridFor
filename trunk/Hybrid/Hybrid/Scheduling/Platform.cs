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
            ComputeDevices.AddRange(Gpu.GpuComputeDevice.GpuComputeDevices());
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
            ComputeDevices.AddRange(Cpu.CpuComputeDevice.CpuComputeDevices());
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
