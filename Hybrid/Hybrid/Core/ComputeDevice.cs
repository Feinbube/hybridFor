using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;
using System.Runtime.InteropServices;
using Hybrid.Core;

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
        public List<MemoryInfo> Caches;

        public List<ComputeUnit> ComputeUnits;

        public bool isBusy = false;

        public Workload CurrentWorkload { get { return History.Count > 0 ? History[History.Count - 1] : null; } }
        public List<Workload> History = new List<Workload>();

        public double PredictPerformance(AlgorithmCharacteristics algorithmCharacteristics)
        {
            double result = 0.0;

            foreach (ComputeUnit computeUnit in ComputeUnits)
                result += computeUnit.PredictPerformance(algorithmCharacteristics);

            // TODO: consider DeviceTypes
            // TODO: consider Memory

            return result;
        }

        public void ParallelFor(int fromInclusive, int toExclusive, Action<int> action)
        {
            Workload workload = new Workload(fromInclusive, toExclusive, action);
            History.Add(workload);

            workload.Start();
            parallelFor(fromInclusive, toExclusive, action);
            workload.Finish();
        }

        public void ParallelFor(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            Workload workload = new Workload(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
            History.Add(workload);

            workload.Start();
            parallelFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
            workload.Finish();
        }

        abstract protected void parallelFor(int fromInclusive, int toExclusive, Action<int> action);
        abstract protected void parallelFor(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action);
    }
}