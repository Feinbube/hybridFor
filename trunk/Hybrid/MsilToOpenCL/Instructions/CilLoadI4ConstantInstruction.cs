using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilLoadI4ConstantInstruction : CilInstruction
    {
        private int m_Constant;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int? Constant = null;
            if (Opcode == OpCodes.Ldc_I4)
            {
                Constant = ReadInt32(IL, Offset + Opcode.Size);
            }
            else if (Opcode == OpCodes.Ldc_I4_S)
            {
                Constant = ReadInt8(IL, Offset + Opcode.Size);
            }

            return new CilLoadI4ConstantInstruction(Opcode, Offset, Constant);
        }

        private CilLoadI4ConstantInstruction(OpCode Opcode, int Offset, int? Constant)
            : base(Opcode, Offset)
        {
            if (Opcode == OpCodes.Ldc_I4_0)
            {
                Constant = 0;
            }
            else if (Opcode == OpCodes.Ldc_I4_1)
            {
                Constant = 1;
            }
            else if (Opcode == OpCodes.Ldc_I4_2)
            {
                Constant = 2;
            }
            else if (Opcode == OpCodes.Ldc_I4_3)
            {
                Constant = 3;
            }
            else if (Opcode == OpCodes.Ldc_I4_4)
            {
                Constant = 4;
            }
            else if (Opcode == OpCodes.Ldc_I4_5)
            {
                Constant = 5;
            }
            else if (Opcode == OpCodes.Ldc_I4_6)
            {
                Constant = 6;
            }
            else if (Opcode == OpCodes.Ldc_I4_7)
            {
                Constant = 7;
            }
            else if (Opcode == OpCodes.Ldc_I4_8)
            {
                Constant = 8;
            }
            else if (Opcode == OpCodes.Ldc_I4_M1)
            {
                Constant = -1;
            }
            System.Diagnostics.Debug.Assert(Constant.HasValue);
            m_Constant = Constant.Value;
        }

        public override string ToString()
        {
            return base.ToString() + (Opcode == OpCodes.Ldc_I4 || Opcode == OpCodes.Ldc_I4_S ? " " + m_Constant.ToString() : string.Empty);
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), m_Constant);
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1, typeof(int)), new HighLevel.IntegerConstantNode(m_Constant)));
            return List;
        }
    }
}
