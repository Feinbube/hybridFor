using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Core
{
    public class Workload
    {
        public enum Dimensions { None, OneDimensional, TwoDimensional }

        object action;

        int fromInclusiveX;
        int toExclusiveX;
        int fromInclusiveY;
        int toExclusiveY;

        DateTime started;
        DateTime finished;
        TimeSpan runtime;

        public Dimensions Dimension
        {
            get
            {
                if (toExclusiveX > fromInclusiveX)
                    if (toExclusiveY > fromInclusiveY)
                        return Dimensions.TwoDimensional;
                    else
                        return Dimensions.OneDimensional;
                else
                    return Dimensions.None;
            }
        }

        public long WorkItemCount
        {
            get
            {
                if (Dimension == Dimensions.None)
                    return 0;
                if (Dimension == Dimensions.OneDimensional)
                    return toExclusiveX - fromInclusiveX;
                if (Dimension == Dimensions.TwoDimensional)
                    return (toExclusiveX - fromInclusiveX) * (toExclusiveY - fromInclusiveY);

                throw new Exception("Dimension " + Dimension + " is unknown!");
            }
        }

        public bool Valid { get { return WorkItemCount > 0; } }

        public double AverageRuntimeInMilliseconds
        {
            get
            {
                if (!Valid || runtime == null)
                    return 0;
                else
                    return runtime.TotalMilliseconds / WorkItemCount;
            }
        }

        public Workload(int fromInclusive, int toExclusive, Action<int> action)
        {
            this.action = action;

            this.fromInclusiveX = fromInclusive;
            this.toExclusiveX = toExclusive;
        }

        public Workload(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            this.action = action;

            this.fromInclusiveX = fromInclusiveX;
            this.toExclusiveX = toExclusiveX;

            this.fromInclusiveY = fromInclusiveY;
            this.toExclusiveY = toExclusiveY;
        }

        public void Start()
        {
            started = DateTime.Now;
        }

        public void Finish()
        {
            finished = DateTime.Now;
            runtime = finished - started;
        }
    }
}