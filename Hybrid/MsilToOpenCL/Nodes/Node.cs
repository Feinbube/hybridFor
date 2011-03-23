using System;
using System.Collections.Generic;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public abstract class Node
    {
        private NodeType m_NodeType;
        private Type m_DataType;
        private List<Node> m_SubNodes;
        private HlGraph m_HlGraph;

        protected Node(NodeType NodeType, Type DataType, bool IsLeaf)
        {
            m_NodeType = NodeType;
            m_DataType = DataType;
            m_SubNodes = new List<Node>();
        }

        public HlGraph HlGraph { get { return m_HlGraph; } set { m_HlGraph = value; } }
        public NodeType NodeType { get { return m_NodeType; } }
        public virtual Type DataType { get { return m_DataType; } set { m_DataType = value; } }
        public List<Node> SubNodes { get { return m_SubNodes; } }
    }
}
