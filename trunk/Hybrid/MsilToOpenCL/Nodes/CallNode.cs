using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class CallNode : Node
    {
        private System.Reflection.MethodInfo m_MethodInfo;
        private bool m_IsStaticCall;

        public CallNode(System.Reflection.MethodInfo MethodInfo)
            : base(NodeType.Call, MethodInfo.ReturnType, false)
        {
            m_MethodInfo = MethodInfo;
            m_IsStaticCall = !object.ReferenceEquals(MethodInfo, null) && ((MethodInfo.CallingConvention & System.Reflection.CallingConventions.HasThis) == 0);
        }

        public CallNode(System.Reflection.MethodInfo MethodInfo, params Node[] Arguments)
            : this(MethodInfo)
        {
            SubNodes.AddRange(Arguments);
        }

        public System.Reflection.MethodInfo MethodInfo { get { return m_MethodInfo; } }
        public bool IsStaticCall { get { return m_IsStaticCall; } set { m_IsStaticCall = value; } }

        public override string ToString()
        {
            StringBuilder String = new StringBuilder();
            int i = 0;

            if (!IsStaticCall)
            {
                if (SubNodes.Count == 0)
                {
                    String.Append("(???).");
                }
                else
                {
                    String.Append(SubNodes[0].ToString());
                    String.Append(".");
                }
                i++;
            }

            string Name = object.ReferenceEquals(HlGraph, null) ? OpenClAliasAttribute.Get(MethodInfo) : HlGraph.GetOpenClFunctionName(MethodInfo);
            if (Name == null)
            {
                Name = MethodInfo.Name;
            }
            String.Append(Name);
            String.Append("(");

            bool IsFirst = true;
            for (; i < SubNodes.Count; i++)
            {
                if (IsFirst)
                {
                    IsFirst = false;
                }
                else
                {
                    String.Append(", ");
                }
                String.Append(SubNodes[i].ToString());
            }

            String.Append(")");

            return String.ToString();
        }
    }
}
