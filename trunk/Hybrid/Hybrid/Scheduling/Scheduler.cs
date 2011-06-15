using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid
{
    public class Scheduler
    {
        public static Platform Platform = new Platform();

        public static void AutomaticFor(int fromInclusive, int toExclusive, Action<int> action)
        {
            ExecuteEvenDistributed(fromInclusive, toExclusive, action);
        }

        private static void ExecuteEvenDistributed(int fromInclusive, int toExclusive, Action<int> action)
        {
            List<ExecuteInfo> executeInfos = new List<ExecuteInfo>();

            int count = toExclusive - fromInclusive;
            int computeDeviceCount = Platform.ComputeDevices.Count;
            int workshare = count / computeDeviceCount;
            int moreWorkCount = count - workshare * computeDeviceCount;

            AlgorithmCharacteristics algorithmCharacteristics = new AlgorithmCharacteristics(action);
            // TODO: do something clever with algochars

            for (int i = 0; i < moreWorkCount; i++)
            {
                int from = fromInclusive + i * (workshare + 1);
                int to = from + (workshare + 1);
                executeInfos.Add(new ExecuteInfo(from, to, action, Platform.ComputeDevices[i]));
            }

            fromInclusive = fromInclusive + moreWorkCount * (workshare + 1);

            for (int i = 0; i < computeDeviceCount - moreWorkCount; i++)
            {
                int from = fromInclusive + i * workshare;
                int to = from + workshare;
                executeInfos.Add(new ExecuteInfo(from, to, action, Platform.ComputeDevices[i]));
            }
            
            System.Threading.Tasks.Parallel.ForEach(executeInfos, executeInfo => { executeInfo.Execute(); });
        }

        public static void AutomaticFor(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            ExecuteEvenDistributedByColumn(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
        }

        private static void ExecuteEvenDistributedByColumn(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int,int> action)
        {
            int count = toExclusiveX - fromInclusiveX;
            int computeDeviceCount = Platform.ComputeDevices.Count;
            int workshare = count / computeDeviceCount;
            int moreWorkCount = count - workshare * computeDeviceCount;

            for (int i = 0; i < moreWorkCount; i++)
            {
                int from = fromInclusiveX + i * (workshare + 1);
                int to = from + (workshare + 1);
                execute(from, to, fromInclusiveY, toExclusiveY, action, Platform.ComputeDevices[i]);
            }

            fromInclusiveX = fromInclusiveX + moreWorkCount * (workshare + 1);

            for (int i = 0; i < computeDeviceCount - moreWorkCount; i++)
            {
                int from = fromInclusiveX + i * workshare;
                int to = from + workshare;
                execute(from, to, fromInclusiveY, toExclusiveY, action, Platform.ComputeDevices[i]);
            }
        }

        private static void execute(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int,int> action, ComputeDevice computeDevice)
        {
            computeDevice.ParallelFor(fromInclusiveX, toExclusiveX, fromInclusiveY, toExclusiveY, action);
        }

        public static void AutomaticInvoke(Action[] actions)
        {
            // Todo Implement me
            foreach (Action action in actions)
                action();
        }
    }
}