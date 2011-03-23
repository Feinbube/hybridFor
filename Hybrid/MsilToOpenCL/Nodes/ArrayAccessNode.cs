using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class ArrayAccessNode : Node
    {
        private System.Type m_ArrayType;

        public ArrayAccessNode(Type ArrayType)
            : base(NodeType.ArrayAccess, ArrayType.GetElementType(), false)
        {
            if (!ArrayType.IsArray)
            {
                throw new ArgumentException("ArrayAccessNode requires an array type.");
            }

            m_ArrayType = ArrayType;
        }

        public Type ArrayType { get { return m_ArrayType; } }

        public override string ToString()
        {
            if (SubNodes.Count == 0)
            {
                return "(???)[ ??? ]";
            }

            StringBuilder String = new StringBuilder();
            String.Append(SubNodes[0].ToString());
            String.Append("[");

            if (SubNodes.Count == 1)
            {
                String.Append(" ??? ]");
            }
            else
            {
                for (int i = 1; i < SubNodes.Count; i++)
                {
                    if (i > 1)
                    {
                        String.Append(", ");
                    }
                    String.Append(SubNodes[i].ToString());
                }
                String.Append("]");
            }

            return String.ToString();
        }

        internal void FlattenArrayType()
        {
            if (ArrayType.GetArrayRank() > 1)
            {
                m_ArrayType = System.Array.CreateInstance(ArrayType.GetElementType(), 1).GetType();
            }
        }
    }
}
