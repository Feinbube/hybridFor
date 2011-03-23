using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class LessNode : BinaryComparisonOperatorNode
    {
        public LessNode(Node Left, Node Right)
            : base(NodeType.Less, Left, Right)
        {
        }

        public override string Symbol
        {
            get { return "<"; }
        }
    }
}
