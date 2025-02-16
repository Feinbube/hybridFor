/*    
*    ComputeUnit.cs
*
﻿*    Copyright (C) 2012  Frank Feinbube, Jan-Arne Sobania, Ralf Diestelkämper
*
*    This library is free software: you can redistribute it and/or modify
*    it under the terms of the GNU Lesser General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public License
*    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*    Frank [at] Feinbube [dot] de
*    jan-arne [dot] sobania [at] gmx [dot] net
*    ralf [dot] diestelkaemper [at] hotmail [dot] com
*
*/


﻿using System;
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

        public int RegisterCount;

        public MemoryInfo SharedMemory;
        public List<MemoryInfo> Caches;

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
