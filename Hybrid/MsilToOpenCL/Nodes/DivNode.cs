using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class DivNode : BinaryOperatorNode
    {
        public DivNode(Node Left, Node Right)
            : base(NodeType.Div, Left, Right)
        {
        }

        public override string Symbol { get { return "/"; } }
    }
}
