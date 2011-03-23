using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class NegNode : UnaryOperatorNode
    {
        public NegNode(Node Argument)
            : base(NodeType.Neg, Argument)
        {
        }

        public override string Symbol { get { return "-"; } }
    }
}
