using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilBranchInstruction : CilInstruction
    {
        private int m_BranchTargetOffset;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int BranchTarget;
            if (Opcode.OperandType == OperandType.ShortInlineBrTarget)
            {
                BranchTarget = NextOffset + ReadInt8(IL, Offset + Opcode.Size);
            }
            else if (Opcode.OperandType == OperandType.InlineBrTarget)
            {
                BranchTarget = NextOffset + ReadInt32(IL, Offset + Opcode.Size);
            }
            else
            {
                throw new ArgumentException("Opcode \"" + Opcode.ToString() + "\" invalid for CilConditionalBranchInstruction.");
            }

            return new CilBranchInstruction(Opcode, Offset, BranchTarget);
        }

        private CilBranchInstruction(OpCode Opcode, int Offset, int BranchTargetOffset)
            : base(Opcode, Offset)
        {
            m_BranchTargetOffset = BranchTargetOffset;
        }

        public override bool CanFallThrough
        {
            get
            {
                return false;
            }
        }

        public override IEnumerable<int> BranchTargetOffsets
        {
            get
            {
                yield return m_BranchTargetOffset;
            }
        }

        public override string ToString()
        {
            return base.ToString() + " IL_" + m_BranchTargetOffset.ToString("X4");
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}goto {1};", IndentString(indent), LabelName(m_BranchTargetOffset));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.BranchInstruction(Context.GetBlock(m_BranchTargetOffset)));
            return List;
        }
    }
}
