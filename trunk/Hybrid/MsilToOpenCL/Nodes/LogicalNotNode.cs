using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class LogicalNotNode : UnaryOperatorNode
    {
        public LogicalNotNode(Node Argument)
            : base(NodeType.LogicalNot, Argument)
        {
        }

        public override string Symbol { get { return "!"; } }
    }
}
