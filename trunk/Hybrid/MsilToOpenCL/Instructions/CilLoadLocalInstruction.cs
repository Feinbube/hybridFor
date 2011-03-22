using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilLoadLocalInstruction : CilInstruction
    {
        private int m_Index;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int Index;
            if (Opcode == OpCodes.Ldloc_S)
            {
                Index = ReadUInt8(IL, Offset + Opcode.Size);
            }
            else if (Opcode == OpCodes.Ldloc)
            {
                Index = ReadUInt16(IL, Offset + Opcode.Size);
            }
            else
            {
                Index = -1;
            }

            return new CilLoadLocalInstruction(Opcode, Offset, Index);
        }

        private CilLoadLocalInstruction(OpCode Opcode, int Offset, int Index)
            : base(Opcode, Offset)
        {
            if (Opcode == OpCodes.Ldloc_0)
            {
                m_Index = 0;
            }
            else if (Opcode == OpCodes.Ldloc_1)
            {
                m_Index = 1;
            }
            else if (Opcode == OpCodes.Ldloc_2)
            {
                m_Index = 2;
            }
            else if (Opcode == OpCodes.Ldloc_3)
            {
                m_Index = 3;
            }
            else if (Opcode == OpCodes.Ldloc || Opcode == OpCodes.Ldloc_S)
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
            return base.ToString() + ((Opcode == OpCodes.Ldloc || Opcode == OpCodes.Ldloc_S) ? " " + m_Index : string.Empty);
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), "local_" + m_Index.ToString());
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1), Context.LocalVariableNode(m_Index)));
            return List;
        }
    }
}
