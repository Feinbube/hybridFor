using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL
{
    internal class CallContext : IDisposable
    {
        public OpenCLNet.Context Context;
        public OpenCLNet.CommandQueue CommandQueue;
        public OpenCLNet.Kernel Kernel;

        public CallContext(OpenCLNet.Context Context, OpenCLNet.Device Device, OpenCLNet.CommandQueueProperties CqProperties, OpenCLNet.Kernel Kernel)
        {
            this.Context = Context;
            this.CommandQueue = Context.CreateCommandQueue(Device, CqProperties);
            this.Kernel = Kernel;
        }

        ~CallContext()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (Kernel != null)
            {
                Kernel.Dispose();
                Kernel = null;
            }
            if (CommandQueue != null)
            {
                CommandQueue.Dispose();
                CommandQueue = null;
            }
            System.GC.SuppressFinalize(this);
        }
    }
}
