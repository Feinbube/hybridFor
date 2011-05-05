using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;
using System.Runtime.InteropServices;

namespace Hybrid
{
    abstract public class ComputeDevice
    {
        public enum DeviceTypes { Cpu, Gpu, Accelerator, Unknown }
        public DeviceTypes DeviceType;

        public string Name;
        public string Manufacturer;
        public string DeviceId;

        public MemoryInfo GlobalMemory;

        public List<ComputeUnit> ComputeUnits;

        

        public double PredictPerformance(AlgorithmCharacteristics algorithmCharacteristics)
        {
            double result = 0.0;

            foreach (ComputeUnit computeUnit in ComputeUnits)
                result += computeUnit.PredictPerformance(algorithmCharacteristics);

            // TODO: consider DeviceTypes
            // TODO: consider Memory

            return result;
        }

        abstract public void ParallelFor(int fromInclusive, int toExclusive, Action<int> action);
        abstract public void ParallelFor(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int,int> action);
    }
}