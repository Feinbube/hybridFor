using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class AlgorithmCharacteristics
    {
        private Action<int> action;

        public AlgorithmCharacteristics()
        {
        }

        public AlgorithmCharacteristics(Action<int> action)
        {
            // atomic attribute?
            // double?

            this.action = action;
        }

        public bool UsesDoublePrecisionFloatingPoint { get; set; }
        public bool UsesAtomics { get; set; }
    }
}
