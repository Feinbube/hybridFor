using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class IntegerConstantNode : Node
    {
        private ulong m_Value;

        public IntegerConstantNode(int Value)
            : base(NodeType.IntegerConstant, typeof(int), true)
        {
            m_Value = (ulong)(long)Value;
        }

        public IntegerConstantNode(uint Value)
            : base(NodeType.IntegerConstant, typeof(uint), true)
        {
            m_Value = (ulong)Value;
        }

        public IntegerConstantNode(long Value)
            : base(NodeType.IntegerConstant, typeof(long), true)
        {
            m_Value = (ulong)Value;
        }

        public IntegerConstantNode(ulong Value)
            : base(NodeType.IntegerConstant, typeof(ulong), true)
        {
            m_Value = Value;
        }

        public object Value
        {
            get
            {
                return m_Value;
            }
        }

        public override string ToString()
        {
            return m_Value.ToString();
        }
    }
}
