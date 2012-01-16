using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class AndNode : BinaryOperatorNode
    {
        public AndNode(Node Left, Node Right)
            : base(NodeType.And, Left, Right)
        {
        }

        public override string Symbol { get { return "&"; } }
    }
}
