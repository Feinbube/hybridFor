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

        public bool ContainsAGpu { get { return deviceOfTypeInDeviceList(ComputeDevice.DeviceTypes.Gpu); } }

        public Platform()
        {
            findOpenCLDevices();

            if (!deviceOfTypeInDeviceList(ComputeDevice.DeviceTypes.Cpu))
                findProcessors();
        }

        private void findOpenCLDevices()
        {
			foreach(Hybrid.Gpu.GpuComputeDevice d in Gpu.GpuComputeDevice.GpuComputeDevices())
				ComputeDevices.Add(d);
        }

        private bool deviceOfTypeInDeviceList(ComputeDevice.DeviceTypes type)
        {
            foreach (ComputeDevice computeDevice in ComputeDevices)
                if (computeDevice.DeviceType == type)
                    return true;

            return false;
        }

        private void findProcessors()
        {
			foreach (Hybrid.Cpu.CpuComputeDevice d in Cpu.CpuComputeDevice.CpuComputeDevices()) {
				ComputeDevices.Add(d);
			}
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
