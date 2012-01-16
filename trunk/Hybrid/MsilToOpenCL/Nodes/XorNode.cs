using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class XorNode : BinaryOperatorNode
    {
        public XorNode(Node Left, Node Right)
            : base(NodeType.Xor, Left, Right)
        {
        }

        public override string Symbol { get { return "^"; } }
    }
}
