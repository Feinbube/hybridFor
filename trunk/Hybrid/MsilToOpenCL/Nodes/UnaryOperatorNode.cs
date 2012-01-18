using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public abstract class UnaryOperatorNode : Node
    {
        public UnaryOperatorNode(NodeType NodeType, Node Argument)
            : base(NodeType, null, false)
        {
            SubNodes.Add(Argument);

            this.DataType = GetResultType(Argument);
        }

        public override string ToString()
        {
            if (SubNodes.Count == 0)
            {
                return Symbol + "(???)";
            }
            else if (SubNodes.Count == 1)
            {
                return "(" + Symbol + SubNodes[0].ToString() + ")";
            }
            else
            {
                return "(??? <too many childs for unary operator node> ???)";
            }
        }

        public virtual Type GetResultType(Node Argument)
        {
            if (Argument == null || Argument.DataType == null)
            {
                return null;
            }

            Type Type = Argument.DataType;

            if (Type == typeof(int) || Type == typeof(long) || Type == typeof(IntPtr) || Type == typeof(float) || Type == typeof(double))
            {
                return Type;
            }

            // TODO
            return null;
        }

        public abstract string Symbol { get; }
    }
}
