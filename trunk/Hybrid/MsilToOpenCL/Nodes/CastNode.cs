using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class CastNode : Node
    {
        private System.Type m_Type;

        public CastNode(Node Argument, Type Type)
            : base(NodeType.Cast, Type, false)
        {
            m_Type = Type;
            SubNodes.Add(Argument);
        }

        public Type Type { get { return m_Type; } }

        public override string ToString()
        {
            return "((" + OpenCLInterop.GetOpenClType(this.HlGraph, m_Type) + ")(" + (SubNodes.Count == 0 ? "???" : SubNodes[0].ToString()) + "))";
        }
    }
}
