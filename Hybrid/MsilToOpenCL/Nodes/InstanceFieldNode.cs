using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class InstanceFieldNode : Node
    {
        private System.Reflection.FieldInfo m_FieldInfo;

        public InstanceFieldNode(Node InstanceNode, System.Reflection.FieldInfo FieldInfo)
            : base(NodeType.InstanceField, FieldInfo.FieldType, false)
        {
            SubNodes.Add(InstanceNode);
            m_FieldInfo = FieldInfo;
        }

        public System.Reflection.FieldInfo FieldInfo { get { return m_FieldInfo; } }

        public override string ToString()
        {
            if (SubNodes.Count == 0)
            {
                return "(???).__field_ref[\"" + (FieldInfo == null ? "???" : FieldInfo.ToString()) + "\"]";
            }
            else if (SubNodes.Count == 1)
            {
                return SubNodes[0].ToString() + "." + ((FieldInfo == null) ? "???" : FieldInfo.Name);
            }
            else
            {
                return "(??? <too many childs for instance field access node> ???)";
            }
        }
    }
}
