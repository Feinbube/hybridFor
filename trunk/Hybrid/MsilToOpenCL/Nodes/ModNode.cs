using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class ModNode : BinaryOperatorNode
    {
        public ModNode(Node Left, Node Right)
            : base(NodeType.Mod, Left, Right)
        {
        }

        public override string Symbol { get { return "%"; } }
    }
}
