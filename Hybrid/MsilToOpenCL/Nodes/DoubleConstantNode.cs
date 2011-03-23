using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class DoubleConstantNode : Node
    {
        private double m_Value;

        public DoubleConstantNode(double Value)
            : base(NodeType.DoubleConstant, typeof(double), true)
        {
            m_Value = Value;
        }

        public double Value
        {
            get
            {
                return m_Value;
            }
        }

        public override string ToString()
        {
            return "((double)" + m_Value.ToString("r", System.Globalization.NumberFormatInfo.InvariantInfo) + ")";
        }
    }
}
