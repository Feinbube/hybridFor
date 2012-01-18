using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilLoadLocalAddressInstruction : CilInstruction
    {
        private int m_Index;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int Index;
            if (Opcode == OpCodes.Ldloca_S)
            {
                Index = ReadUInt8(IL, Offset + Opcode.Size);
            }
            else if (Opcode == OpCodes.Ldloca)
            {
                Index = ReadUInt16(IL, Offset + Opcode.Size);
            }
            else
            {
                Index = -1;
            }

            return new CilLoadLocalAddressInstruction(Opcode, Offset, Index);
        }

        private CilLoadLocalAddressInstruction(OpCode Opcode, int Offset, int Index)
            : base(Opcode, Offset)
        {
            if (Opcode == OpCodes.Ldloca || Opcode == OpCodes.Ldloca_S)
            {
                m_Index = Index;
            }
            else
            {
                throw new ArgumentException("Opcode");
            }

            System.Diagnostics.Debug.Assert(m_Index >= 0);
        }

        public override string ToString()
        {
            return base.ToString() + " " + m_Index;
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = &{2};", IndentString(indent), StackName(CurStack + 1), "local_" + m_Index.ToString());
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1), new HighLevel.AddressOfNode(Context.LocalVariableNode(m_Index))));
            return List;
        }
    }
}
