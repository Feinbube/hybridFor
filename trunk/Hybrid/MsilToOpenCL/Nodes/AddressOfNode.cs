using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class AddressOfNode : UnaryOperatorNode
    {
        public AddressOfNode(Node Argument)
            : base(NodeType.AddressOf, Argument)
        {
        }

        public override string Symbol { get { return "&"; } }

        public override Type GetResultType(Node Argument)
        {
            if (Argument == null || Argument.DataType == null)
            {
                return null;
            }

            Type Type = Argument.DataType;

            if (!Type.IsPointer && !Type.IsByRef)
            {
                return Type.Assembly.GetType(Type.FullName + "*", true);
            }

            // TODO
            return null;
        }
    }
}
