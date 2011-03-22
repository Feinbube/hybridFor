using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class StackState
    {
        private List<StackLocation> m_StackLocations;
        private bool m_Complete;

        public List<StackLocation> StackLocations { get { return m_StackLocations; } }
        public bool Complete { get { return m_Complete; } set { m_Complete = value; } }

        public StackState(int EntryCount, bool? Complete)
        {
            m_StackLocations = new List<StackLocation>(EntryCount);
            for (int i = 0; i < EntryCount; i++)
            {
                m_StackLocations.Add(CreateStackLocation(i, null));
            }

            m_Complete = (Complete.HasValue ? Complete.Value : (m_StackLocations.Count == 0));
        }

        public StackState(StackState ex)
        {
            m_StackLocations = new List<StackLocation>(ex.StackLocations.Count);
            foreach (StackLocation Location in ex.StackLocations)
            {
                m_StackLocations.Add((StackLocation)Location.Clone());
            }
        }

        public static StackLocation CreateStackLocation(int Index, Type DataType)
        {
            return new StackLocation(Index, DataType);
        }
    }
}
