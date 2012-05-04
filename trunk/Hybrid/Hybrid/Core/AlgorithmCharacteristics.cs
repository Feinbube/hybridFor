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

		private bool usesDoublePrecisionFloatingPoint;
		public bool UsesDoublePrecisionFloatingPoint { get { return usesDoublePrecisionFloatingPoint; } set { usesDoublePrecisionFloatingPoint = value; } }

		private bool usesAtomics;
		public bool UsesAtomics { get { return usesAtomics; } set { usesAtomics = value; } }
    }
}
