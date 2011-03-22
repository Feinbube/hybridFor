using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilLoadArgumentInstruction : CilInstruction
    {
        private int m_Index;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int Index;
            if (Opcode == OpCodes.Ldarg_S)
            {
                Index = IL[Offset + Opcode.Size + 0];
            }
            else if (Opcode == OpCodes.Ldarg)
            {
                Index = ReadUInt16(IL, Offset + Opcode.Size);
            }
            else
            {
                Index = -1;
            }

            return new CilLoadArgumentInstruction(Opcode, Offset, Index);
        }

        private CilLoadArgumentInstruction(OpCode Opcode, int Offset, int Index)
            : base(Opcode, Offset)
        {
            if (Opcode == OpCodes.Ldarg_0)
            {
                m_Index = 0;
            }
            else if (Opcode == OpCodes.Ldarg_1)
            {
                m_Index = 1;
            }
            else if (Opcode == OpCodes.Ldarg_2)
            {
                m_Index = 2;
            }
            else if (Opcode == OpCodes.Ldarg_3)
            {
                m_Index = 3;
            }
            else if (Opcode == OpCodes.Ldarg || Opcode == OpCodes.Ldarg_S)
            {
                m_Index = Index;
            }
            else
            {
                throw new ArgumentException();
            }

            System.Diagnostics.Debug.Assert(m_Index >= 0);
        }

        public override string ToString()
        {
            return base.ToString() + ((Opcode == OpCodes.Ldarg || Opcode == OpCodes.Ldarg_S) ? " " + m_Index : string.Empty);
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), "argument_" + m_Index.ToString());
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1), Context.ArgumentNode(m_Index)));
            return List;
        }
    }
}
