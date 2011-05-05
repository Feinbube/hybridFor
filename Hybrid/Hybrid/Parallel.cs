using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class Parallel
    {
        public static ExecutionMode Mode = ExecutionMode.TaskParallel;

        public static void For(int fromInclusive, int toExclusive, Action<int> action)
        {
            For(fromInclusive, toExclusive, action, Mode);
        }

        public static void For(int fromInclusive, int toExclusive, Action<int> action, ExecutionMode mode)
        {
            if (fromInclusive >= toExclusive)
                return;

            switch (mode)
            {
                case ExecutionMode.TaskParallel:
                case ExecutionMode.TaskParallel2D:
                    Cpu.CpuComputeDevice.CpuParallelFor(fromInclusive, toExclusive, action);
                    break;

                case ExecutionMode.Gpu:
                case ExecutionMode.Gpu2D:
                    Gpu.GpuComputeDevice.GpuParallelFor(fromInclusive, toExclusive, action);
                    break;

                case ExecutionMode.Serial:
                    for (int i = fromInclusive; i < toExclusive; i++)
                        action(i);
                    break;

                case ExecutionMode.Automatic:
                    Scheduler.AutomaticFor(fromInclusive, toExclusive, action);
                    break;

                default:
                    throw new NotImplementedException("Execution mode " + mode.ToString() + " is not supported.");
            }
        }

        public static void For(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            For(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action, Mode);
        }

        public static void For(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action, ExecutionMode mode)
        {
            if (fromInclusiveX >= toExclusiveX || fromInclusiveY >= toExclusiveY)
                return;

            switch (mode)
            {
                case ExecutionMode.TaskParallel:
                    Cpu.CpuComputeDevice.CpuParallelFor(fromInclusiveX, toExclusiveX, delegate(int x)
                    {
                        for (int y = fromInclusiveY; y < toExclusiveY; y++)
                            action(x, y);
                    });
                    break;

                case ExecutionMode.TaskParallel2D:
                    Cpu.CpuComputeDevice.CpuParallelFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
                    break;

                case ExecutionMode.Gpu:
                // TODO: Uncomment me
                //if (mode == ExecutionMode.Gpu)
                //    Gpu.Parallel.For(fromInclusiveX, toExclusiveX, delegate(int x)
                //    {
                //        for (int y = fromInclusiveY; y < toExclusiveY; y++)
                //            action(x, y);
                //    });
                //    break;
                case ExecutionMode.Gpu2D:
                    Gpu.GpuComputeDevice.GpuParallelFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
                    break;

                case ExecutionMode.Serial:
                    for (int x = fromInclusiveX; x < toExclusiveX; x++)
                        for (int y = fromInclusiveY; y < toExclusiveY; y++)
                            action(x, y);
                    break;

                case ExecutionMode.Automatic:
                    Scheduler.AutomaticFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
                    break;

                default:
                    throw new NotImplementedException("Execution mode " + mode.ToString() + " is not supported.");
            }
        }

        public static void Invoke(params Action[] actions)
        {
            switch (Mode)
            {
                case ExecutionMode.TaskParallel:
                case ExecutionMode.TaskParallel2D:
                    System.Threading.Tasks.Parallel.Invoke(actions);
                    break;

                case ExecutionMode.Gpu:
                case ExecutionMode.Gpu2D:
                    Gpu.GpuComputeDevice.GpuParallelInvoke(actions);
                    break;

                case ExecutionMode.Serial:
                    foreach (Action action in actions)
                        action();
                    break;

                case ExecutionMode.Automatic:
                    Scheduler.AutomaticInvoke(actions);
                    break;

                default:
                    throw new NotImplementedException("Execution mode " + Mode.ToString() + " is not supported.");
            }
        }

        public static void ReInitialize()
        {
            if (Mode == ExecutionMode.Gpu || Mode == ExecutionMode.Gpu2D)
                Gpu.GpuComputeDevice.GpuReInitialize();
        }
    }
}
