﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Gpu
{
    public class Atomic
    {
        // this function is NEVER called because it will be replaced in Hyrid.Parallel regarding the annotation at the base class!!
        public static int Add(ref int location1, int value)
        {
            // implement me // OpenCL-Spec 6.11.11 atomic_add
            return System.Threading.Interlocked.Add(ref location1, value);
        }
    }
}