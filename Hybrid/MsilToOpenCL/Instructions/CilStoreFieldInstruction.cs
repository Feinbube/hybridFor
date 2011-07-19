using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Reflection.Emit;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilStoreFieldInstruction : CilInstruction
    {
        private FieldInfo m_FieldInfo;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int Token = ReadInt32(IL, Offset + Opcode.Size);
            FieldInfo FieldInfo = ParentMethodBase.Module.ResolveField(Token);
            return new CilStoreFieldInstruction(Opcode, Offset, FieldInfo);
        }

        private CilStoreFieldInstruction(OpCode Opcode, int Offset, FieldInfo FieldInfo)
            : base(Opcode, Offset)
        {
            if (Opcode != OpCodes.Stfld)
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
            writer.WriteLine("{0}({1}).__field(\"{2}\") = {3};", IndentString(indent), StackName(CurStack - 1), m_FieldInfo.DeclaringType.FullName + "::" + m_FieldInfo.Name, StackName(CurStack));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

            HighLevel.Node Result;
            if (FieldInfo.IsStatic)
            {
                Result = new HighLevel.LocationNode(Context.StaticFieldLocation(FieldInfo));
            }
            else
            {
                Result = new HighLevel.InstanceFieldNode(Context.ReadStackLocationNode(Context.StackPointer - 1), FieldInfo);
            }

            List.Add(new HighLevel.AssignmentInstruction(Result, Context.ReadStackLocationNode(Context.StackPointer)));
            return List;
        }
    }
}
