using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class SubNode : BinaryOperatorNode
    {
        public SubNode(Node Left, Node Right)
            : base(NodeType.Sub, Left, Right)
        {
        }

        public override string Symbol { get { return "-"; } }
    }
}
