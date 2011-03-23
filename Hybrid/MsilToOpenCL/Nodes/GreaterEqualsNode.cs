using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class GreaterEqualsNode : BinaryComparisonOperatorNode
    {
        public GreaterEqualsNode(Node Left, Node Right)
            : base(NodeType.GreaterEquals, Left, Right)
        {
        }

        public override string Symbol
        {
            get { return ">="; }
        }
    }
}
