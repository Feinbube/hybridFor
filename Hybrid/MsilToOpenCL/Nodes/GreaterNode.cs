using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class GreaterNode : BinaryComparisonOperatorNode
    {
        public GreaterNode(Node Left, Node Right)
            : base(NodeType.Greater, Left, Right)
        {
        }

        public override string Symbol
        {
            get { return ">"; }
        }
    }
}
