using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class NamedFieldNode : Node
    {
        private string m_FieldName;

        public NamedFieldNode(Node InstanceNode, string FieldName, Type FieldType)
            : base(NodeType.NamedField, FieldType, false)
        {
            SubNodes.Add(InstanceNode);
            m_FieldName = FieldName;
        }

        public string FieldName { get { return m_FieldName; } }

        public override string ToString()
        {
            if (SubNodes.Count == 0)
            {
                return "(???).__field_ref[\"" + (FieldName == null ? "???" : FieldName) + "\"]";
            }
            else if (SubNodes.Count == 1)
            {
                return SubNodes[0].ToString() + "." + ((FieldName == null) ? "???" : FieldName);
            }
            else
            {
                return "(??? <too many childs for named field access node> ???)";
            }
        }
    }
}
