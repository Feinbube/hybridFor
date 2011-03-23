using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class AddNode : BinaryOperatorNode
    {
        public AddNode(Node Left, Node Right)
            : base(NodeType.Add, Left, Right)
        {
        }

        public override string Symbol { get { return "+"; } }
    }
}
