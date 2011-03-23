using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public abstract class BinaryComparisonOperatorNode : BinaryOperatorNode
    {
        public BinaryComparisonOperatorNode(NodeType NodeType, Node Left, Node Right)
            : base(NodeType, Left, Right)
        {
        }

        public override Type GetResultType(Node Left, Node Right)
        {
            return typeof(int);
        }
    }
}
