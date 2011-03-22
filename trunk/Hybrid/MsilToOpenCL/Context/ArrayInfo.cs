using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class ArrayInfo
    {
        private ArgumentLocation m_ArrayArgument;
        private List<Node> m_ScaleNode;
        private List<ArgumentLocation> m_ScaleArgument;
        private int m_DimensionCount;

        public ArrayInfo(ArgumentLocation ArrayArgument)
        {
            m_ArrayArgument = ArrayArgument;
            m_DimensionCount = ArrayArgument.DataType.GetArrayRank();

            m_ScaleNode = new List<Node>(DimensionCount);
            m_ScaleArgument = new List<ArgumentLocation>(DimensionCount);

            for (int i = 0; i < DimensionCount; i++)
            {
                m_ScaleNode.Add(null);
                m_ScaleArgument.Add(null);
            }
        }

        public int DimensionCount { get { return m_DimensionCount; } }

        public List<Node> ScaleNode { get { return m_ScaleNode; } }
        public List<ArgumentLocation> ScaleArgument { get { return m_ScaleArgument; } }
    }
}
