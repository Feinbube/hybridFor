using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace Hybrid.MsilToOpenCL
{
    public class HlGraphCache
    {
        private Dictionary<IntPtr, Dictionary<MethodInfo, HlGraphEntry>> hlGraphCache = new Dictionary<IntPtr, Dictionary<MethodInfo, HlGraphEntry>>();

        internal void purge()
        {
            lock (hlGraphCache)
            {
                hlGraphCache.Clear();
            }
        }

        internal bool TryGetValue(IntPtr deviceId, MethodInfo methodInfo, out HlGraphEntry hlGraphEntry)
        {
            lock (hlGraphCache)
            {
                if (hlGraphCache.ContainsKey(deviceId) && hlGraphCache[deviceId].TryGetValue(methodInfo, out hlGraphEntry))
                    return true;
                else
                {
                    hlGraphEntry = null;
                    return false;
                }
            }
        }

        internal void SetValue(IntPtr deviceId, MethodInfo methodInfo, HlGraphEntry hlGraphEntry)
        {
            lock (hlGraphCache)
            {
                if (!hlGraphCache.ContainsKey(deviceId))
                    hlGraphCache[deviceId] = new Dictionary<MethodInfo, HlGraphEntry>();

                hlGraphCache[deviceId][methodInfo] = hlGraphEntry;
            }
        }
    }
}
