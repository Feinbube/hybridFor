﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class ExecuteInfo
    {
        int fromInclusive;
        int toExclusive;

        Action<int> action;
        ComputeDevice computeDevice;

        public ExecuteInfo(int fromInclusive, int toExclusive, Action<int> action, ComputeDevice computeDevice)
        {
            this.fromInclusive = fromInclusive;
            this.toExclusive = toExclusive;
            this.action = action;
            this.computeDevice = computeDevice;
        }

        public ExecuteInfo(int fromInclusive, int toExclusive, Action<int> action)
        {
            this.fromInclusive = fromInclusive;
            this.toExclusive = toExclusive;
            this.action = action;
        }

        public void Execute()
        {
            computeDevice.ParallelFor(fromInclusive, toExclusive, action);
        }

        public void Execute(ComputeDevice computeDevice)
        {
            this.computeDevice = computeDevice;
            computeDevice.ParallelFor(fromInclusive, toExclusive, action);
        }
    }
}
