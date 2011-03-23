using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL
{
    internal class HlGraphCacheEntry : IDisposable
    {
        private HighLevel.HlGraph m_HlGraph;
        private List<HighLevel.ArgumentLocation> m_fromInclusiveLocation;
        private List<HighLevel.ArgumentLocation> m_toExclusiveLocation;
        private string m_Source;

        public HlGraphCacheEntry(HighLevel.HlGraph HlGraph, List<HighLevel.ArgumentLocation> fromInclusiveLocation, List<HighLevel.ArgumentLocation> toExclusiveLocation)
        {
            m_HlGraph = HlGraph;
            m_fromInclusiveLocation = fromInclusiveLocation;
            m_toExclusiveLocation = toExclusiveLocation;
        }

        public HighLevel.HlGraph HlGraph { get { return m_HlGraph; } }
        public List<HighLevel.ArgumentLocation> fromInclusiveLocation { get { return m_fromInclusiveLocation; } }
        public List<HighLevel.ArgumentLocation> toExclusiveLocation { get { return m_toExclusiveLocation; } }
        public string Source { get { return m_Source; } set { m_Source = value; } }

        private OpenCLNet.Context m_Context;
        private OpenCLNet.Program m_Program;
        public OpenCLNet.Context Context { get { return m_Context; } set { m_Context = value; } }
        public OpenCLNet.Program Program { get { return m_Program; } set { m_Program = value; } }

        ~HlGraphCacheEntry()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (m_Program != null)
            {
                m_Program.Dispose();
                m_Program = null;
            }
            if (m_Context != null)
            {
                m_Context.Dispose();
                m_Context = null;
            }
            System.GC.SuppressFinalize(this);
        }
    }
}
