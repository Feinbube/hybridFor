using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;

namespace Hybrid
{
    public class ComputeUnit
    {
        public bool AtomicsSupported;
        public bool DoublePrecisionFloatingPointSupported;

        public MemoryInfo SharedMemory;
        public MemoryInfo GlobalMemoryCache;

        public List<ProcessingElement> ProcessingElements;

        public double PredictPerformance(AlgorithmCharacteristics algorithmCharacteristics)
        {
            double result = 0.0;

            foreach (ProcessingElement processingElement in ProcessingElements)
                result += processingElement.ClockSpeed;

            if (algorithmCharacteristics.UsesDoublePrecisionFloatingPoint && !DoublePrecisionFloatingPointSupported)
                result /= 8; // TODO: check heuristics

            if (algorithmCharacteristics.UsesAtomics && !AtomicsSupported)
                result = 0;

            // TODO: consider memory as well

            return result;
        }
    }
}
