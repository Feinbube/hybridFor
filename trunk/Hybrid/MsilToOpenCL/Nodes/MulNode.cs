using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class MulNode : BinaryOperatorNode
    {
        public MulNode(Node Left, Node Right)
            : base(NodeType.Mul, Left, Right)
        {
        }

        public override string Symbol { get { return "*"; } }
    }
}
