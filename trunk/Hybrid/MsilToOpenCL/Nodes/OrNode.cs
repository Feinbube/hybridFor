using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class OrNode : BinaryOperatorNode
    {
        public OrNode(Node Left, Node Right)
            : base(NodeType.Or, Left, Right)
        {
        }

        public override string Symbol { get { return "|"; } }
    }
}
