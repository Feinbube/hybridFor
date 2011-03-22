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

            if (mode == ExecutionMode.TaskParallel || mode == ExecutionMode.TaskParallel2D)
                System.Threading.Tasks.Parallel.For(fromInclusive, toExclusive, action);

            if (mode == ExecutionMode.Gpu || mode == ExecutionMode.Gpu2D)
                Gpu.Parallel.For(fromInclusive, toExclusive, action);

            if (mode == ExecutionMode.Serial)
                for (int i = fromInclusive; i < toExclusive; i++)
                    action(i);
        }

        public static void For(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            For(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action, Mode);
        }

        public static void For(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action, ExecutionMode mode)
        {
            if (fromInclusiveX >= toExclusiveX || fromInclusiveY >= toExclusiveY)
            {
                return;
            }

            if (mode == ExecutionMode.TaskParallel)
                System.Threading.Tasks.Parallel.For(fromInclusiveX, toExclusiveX, delegate(int x)
                {
                    for (int y = fromInclusiveY; y < toExclusiveY; y++)
                        action(x, y);
                });

            if (mode == ExecutionMode.TaskParallel2D)
                System.Threading.Tasks.Parallel.For(fromInclusiveX, toExclusiveX, delegate(int x)
                {
                    System.Threading.Tasks.Parallel.For(fromInclusiveY, toExclusiveY, delegate(int y)
                    {
                        action(x, y);
                    });
                });

            // TODO: Uncomment me
            //if (mode == ExecutionMode.Gpu)
            //    Gpu.Parallel.For(fromInclusiveX, toExclusiveX, delegate(int x)
            //    {
            //        for (int y = fromInclusiveY; y < toExclusiveY; y++)
            //            action(x, y);
            //    });

            if (mode == ExecutionMode.Gpu || mode == ExecutionMode.Gpu2D)
                Gpu.Parallel.For(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);

            if (mode == ExecutionMode.Serial)
                for (int x = fromInclusiveX; x < toExclusiveX; x++)
                    for (int y = fromInclusiveY; y < toExclusiveY; y++)
                        action(x, y);
        }

        public static void Invoke(params Action[] actions)
        {
            if (Mode == ExecutionMode.TaskParallel || Mode == ExecutionMode.TaskParallel2D)
                System.Threading.Tasks.Parallel.Invoke(actions);

            if (Mode == ExecutionMode.Gpu || Mode == ExecutionMode.Gpu2D)
                Gpu.Parallel.Invoke(actions);

            if (Mode == ExecutionMode.Serial)
                foreach (Action action in actions)
                    action();
        }

        public static void ReInitialize()
        {
            if (Mode == ExecutionMode.Gpu || Mode == ExecutionMode.Gpu2D)
                Gpu.Parallel.ReInitialize();
        }
    }
}
