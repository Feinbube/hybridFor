using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class NotEqualsNode : BinaryComparisonOperatorNode
    {
        public NotEqualsNode(Node Left, Node Right)
            : base(NodeType.NotEquals, Left, Right)
        {
        }

        public override string Symbol
        {
            get { return "!="; }
        }
    }
}
