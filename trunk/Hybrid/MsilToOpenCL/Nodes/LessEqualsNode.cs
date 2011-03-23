using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class LessEqualsNode : BinaryComparisonOperatorNode
    {
        public LessEqualsNode(Node Left, Node Right)
            : base(NodeType.LessEquals, Left, Right)
        {
        }

        public override string Symbol
        {
            get { return "<="; }
        }
    }
}
