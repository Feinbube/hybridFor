using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public abstract class BinaryOperatorNode : Node
    {
        public BinaryOperatorNode(NodeType NodeType, Node Left, Node Right)
            : base(NodeType, null, false)
        {
            SubNodes.Add(Left);
            SubNodes.Add(Right);

            this.DataType = GetResultType(Left, Right);
        }

        public override string ToString()
        {
            if (SubNodes.Count == 0)
            {
                return "(??? " + Symbol + " ???)";
            }
            else if (SubNodes.Count == 1)
            {
                return "(" + SubNodes[0].ToString() + " " + Symbol + " ???)";
            }
            else if (SubNodes.Count == 2)
            {
                return "(" + SubNodes[0].ToString() + " " + Symbol + " " + SubNodes[1].ToString() + ")";
            }
            else
            {
                return "(??? <too many childs for binary operator node> ???)";
            }
        }

        public virtual Type GetResultType(Node Left, Node Right)
        {
            if (Left == null || Left.DataType == null || Right == null || Right.DataType == null)
            {
                return null;
            }

            Type LeftType = Left.DataType, RightType = Right.DataType;

            if (LeftType == typeof(int))
            {
                if (RightType == typeof(int) || RightType == typeof(IntPtr) || RightType == typeof(uint))
                {
                    return RightType;
                }
            }
            else if (LeftType == typeof(uint))
            {
                if (RightType == typeof(int) || RightType == typeof(uint))
                {
                    return LeftType;
                }
                else if (RightType == typeof(IntPtr))
                {
                    return RightType;
                }
            }
            else if (LeftType == typeof(long))
            {
                if (RightType == LeftType)
                {
                    return RightType;
                }
            }
            else if (LeftType == typeof(IntPtr))
            {
                if (RightType == typeof(int) || RightType == typeof(IntPtr))
                {
                    return RightType;
                }
            }
            else if (LeftType == typeof(float) || LeftType == typeof(double))
            {
                if (RightType == typeof(float) || RightType == typeof(double))
                {
                    return (LeftType == typeof(double) || RightType == typeof(double)) ? typeof(double) : typeof(float);
                }
            }

            // TODO
            return null;
        }

        public abstract string Symbol { get; }
    }
}
