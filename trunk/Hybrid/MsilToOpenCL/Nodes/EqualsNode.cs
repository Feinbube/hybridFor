using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class EqualsNode : BinaryComparisonOperatorNode
    {
        public EqualsNode(Node Left, Node Right)
            : base(NodeType.Equals, Left, Right)
        {
        }

        public override string Symbol
        {
            get { return "=="; }
        }
    }
}
