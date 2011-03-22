using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Reflection.Emit;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilLoadFieldAddressInstruction : CilInstruction
    {
        private FieldInfo m_FieldInfo;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int Token = ReadInt32(IL, Offset + Opcode.Size);
            FieldInfo FieldInfo = ParentMethodBase.Module.ResolveField(Token);
            return new CilLoadFieldAddressInstruction(Opcode, Offset, FieldInfo);
        }

        private CilLoadFieldAddressInstruction(OpCode Opcode, int Offset, FieldInfo FieldInfo)
            : base(Opcode, Offset)
        {
            if (Opcode != OpCodes.Ldflda)
            {
                throw new ArgumentException("Opcode");
            }
            m_FieldInfo = FieldInfo;
        }

        public FieldInfo FieldInfo
        {
            get
            {
                return m_FieldInfo;
            }
        }

        public override string ToString()
        {
            return base.ToString() + " " + m_FieldInfo.DeclaringType.FullName + "::" + m_FieldInfo.Name;
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = &({2}).__field(\"{3}\");", IndentString(indent), StackName(CurStack), StackName(CurStack), m_FieldInfo.DeclaringType.FullName + "::" + m_FieldInfo.Name);
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

            HighLevel.Node Argument;
            if (FieldInfo.IsStatic)
            {
                Argument = new HighLevel.LocationNode(Context.StaticFieldLocation(FieldInfo));
            }
            else
            {
                Argument = new HighLevel.InstanceFieldNode(Context.ReadStackLocationNode(Context.StackPointer), FieldInfo);
            }

            if (!FieldInfo.FieldType.IsValueType)
            {
                Argument = new HighLevel.AddressOfNode(Argument);
            }

            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer), Argument));
            return List;
        }
    }
}
