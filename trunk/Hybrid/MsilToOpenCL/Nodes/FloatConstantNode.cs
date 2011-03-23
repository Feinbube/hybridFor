using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class FloatConstantNode : Node
    {
        private float m_Value;

        public FloatConstantNode(float Value)
            : base(NodeType.FloatConstant, typeof(float), true)
        {
            m_Value = Value;
        }

        public float Value
        {
            get
            {
                return m_Value;
            }
        }

        public override string ToString()
        {
            return "((float)" + m_Value.ToString("r", System.Globalization.NumberFormatInfo.InvariantInfo) + ")";
        }
    }
}
