using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class StringConstantNode : Node
    {
        private string m_Value;

        public StringConstantNode(string Value)
            : base(NodeType.StringConstant, typeof(string), true)
        {
            m_Value = Value;
        }

        public string Value
        {
            get
            {
                return m_Value;
            }
        }

        public override string ToString()
        {
            return "\"" + m_Value + "\"";
        }
    }
}
