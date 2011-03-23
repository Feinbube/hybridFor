using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Hybrid.MsilToOpenCL
{
    public static class OpenClFunctions
    {
        public static void __target_only()
        {
            string Name;
            try
            {
                System.Diagnostics.StackTrace StackTrace = new System.Diagnostics.StackTrace(Thread.CurrentThread, false);
                System.Diagnostics.StackFrame StackFrame = StackTrace.GetFrame(1);
                Name = StackFrame.GetMethod().Name;
            }
            catch (System.Exception ex)
            {
                throw new InvalidOperationException("Sorry, the target function does not have an implementation on the host CPU and can only be invoked from OpenCL.", ex);
            }
            throw new InvalidOperationException(string.Format("Sorry, the function '{0}' does not have an implementation on the host CPU and can only be invoked from OpenCL.", Name));
        }

        [OpenClAlias]
        public static uint get_work_dim()
        {
            __target_only();
            return 0;
        }

        [OpenClAlias]
        public static uint get_global_size(uint dimindx)
        {
            __target_only();
            return 0;
        }

        [OpenClAlias]
        public static uint get_global_id(uint dimindx)
        {
            __target_only();
            return 0;
        }

        [OpenClAlias]
        public static uint get_local_size(uint dimindx)
        {
            __target_only();
            return 0;
        }

        [OpenClAlias]
        public static uint get_local_id(uint dimindx)
        {
            __target_only();
            return 0;
        }

        [OpenClAlias]
        public static uint get_num_groups(uint dimindx)
        {
            __target_only();
            return 0;
        }

        [OpenClAlias]
        public static uint get_group_id(uint dimindx)
        {
            __target_only();
            return 0;
        }
    }
}
