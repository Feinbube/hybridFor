using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class Parallel
    {
        // TODO: uncomment in final version
        //
        //public static void For(int fromInclusive, int toExclusive, Action<int> action)
        //{
        //    For(fromInclusive, toExclusive, action, Execute.OnEverythingAvailable);
        //}

        public static void For(Execute mode, int fromInclusive, int toExclusive, Action<int> action)
        {
            if (fromInclusive >= toExclusive)
                return;

            switch (mode)
            {
                case Execute.OnAllCpus:
                    Cpu.CpuComputeDevice.CpuParallelFor(fromInclusive, toExclusive, action);
                    break;

                case Execute.OnSingleGpu:
                    Gpu.GpuComputeDevice.GpuParallelFor(fromInclusive, toExclusive, action);
                    break;

                case Execute.OnSingleCpu:
                    for (int i = fromInclusive; i < toExclusive; i++)
                        action(i);
                    break;

                case Execute.OnEverythingAvailable:
                    Scheduler.AutomaticFor(fromInclusive, toExclusive, action);
                    break;

                default:
                    throw new NotImplementedException("Execution mode " + mode.ToString() + " is not supported.");
            }
        }

        // TODO: uncomment in final version
        //
        //public static void For(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        //{
        //    For(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action, Execute.OnEverythingAvailable);
        //}

        public static void For(Execute mode, int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            if (fromInclusiveX >= toExclusiveX || fromInclusiveY >= toExclusiveY)
                return;

            switch (mode)
            {
                case Execute.OnAllCpus:
                    Cpu.CpuComputeDevice.CpuParallelFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
                    break;

                case Execute.OnSingleGpu:
                    Gpu.GpuComputeDevice.GpuParallelFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
                    break;

                case Execute.OnSingleCpu:
                    for (int x = fromInclusiveX; x < toExclusiveX; x++)
                        for (int y = fromInclusiveY; y < toExclusiveY; y++)
                            action(x, y);
                    break;

                case Execute.OnEverythingAvailable:
                    Scheduler.AutomaticFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
                    break;

                default:
                    throw new NotImplementedException("Execution mode " + mode.ToString() + " is not supported.");
            }
        }

        public static void Invoke(params Action[] actions)
        {
            Invoke(Execute.OnAllCpus, actions);
        }

        public static void Invoke( Execute mode, params Action[] actions)
        {
            switch (mode)
            {
                case Execute.OnAllCpus:
                    System.Threading.Tasks.Parallel.Invoke(actions);
                    break;

                case Execute.OnSingleGpu:
                    Gpu.GpuComputeDevice.GpuParallelInvoke(actions);
                    break;

                case Execute.OnSingleCpu:
                    foreach (Action action in actions)
                        action();
                    break;

                case Execute.OnEverythingAvailable:
                    Scheduler.AutomaticInvoke(actions);
                    break;

                default:
                    throw new NotImplementedException("Execution mode " + mode.ToString() + " is not supported.");
            }
        }

        public static void ReInitialize()
        {
            Gpu.GpuComputeDevice.GpuReInitialize();
        }
    }
}
