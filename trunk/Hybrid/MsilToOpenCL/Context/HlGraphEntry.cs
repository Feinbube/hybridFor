using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL
{
    internal class HlGraphEntry : IDisposable
    {
        private HighLevel.HlGraph m_HlGraph;

        private List<HighLevel.ArgumentLocation> m_fromInclusiveLocation;
        private List<HighLevel.ArgumentLocation> m_toExclusiveLocation;
        
        private string m_Source;

        public HlGraphEntry(HighLevel.HlGraph HlGraph, List<HighLevel.ArgumentLocation> fromInclusiveLocation, List<HighLevel.ArgumentLocation> toExclusiveLocation)
        {
            m_HlGraph = HlGraph;

            m_fromInclusiveLocation = fromInclusiveLocation;
            m_toExclusiveLocation = toExclusiveLocation;
        }

        public HighLevel.HlGraph HlGraph { get { return m_HlGraph; } }

        public List<HighLevel.ArgumentLocation> fromInclusiveLocation { get { return m_fromInclusiveLocation; } }
        public List<HighLevel.ArgumentLocation> toExclusiveLocation { get { return m_toExclusiveLocation; } }
        
        public string Source { get { return m_Source; } set { m_Source = value; } }

        private OpenCLNet.Context context;
        private OpenCLNet.Program program;
        private OpenCLNet.Device device;

        public OpenCLNet.Context Context { get { return context; } set { context = value; } }
        public OpenCLNet.Program Program { get { return program; } set { program = value; } }
        public OpenCLNet.Device Device { get { return device; } set { device = value; } }

        ~HlGraphEntry()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (program != null)
            {
                program.Dispose();
                program = null;
            }
            if (context != null)
            {
                context.Dispose();
                context = null;
            }
            System.GC.SuppressFinalize(this);
        }
    }
}
