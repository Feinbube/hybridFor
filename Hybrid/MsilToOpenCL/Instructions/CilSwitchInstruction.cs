using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    // TODO support switch on Strings
    class CilSwitchInstruction: CilInstruction
    {
        private List<int> m_BranchTargetOffsets;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int count = ReadInt32(IL, Offset + Opcode.Size);
            List<int> BranchTargets = new List<int>(count);

            for (int i=0; i<count; i++)
                BranchTargets.Add(NextOffset + ReadInt32(IL, Offset + Opcode.Size + (i+1)*4));

            return new CilSwitchInstruction(Opcode, Offset, BranchTargets);
        }

        private CilSwitchInstruction(OpCode Opcode, int Offset, List<int> BranchTargetOffsets)
            : base(Opcode, Offset)
        {
            if (!(Opcode == OpCodes.Switch))
            {
                throw new ArgumentException("Opcode");
            }

            m_BranchTargetOffsets = BranchTargetOffsets;
        }

        public override IEnumerable<int> BranchTargetOffsets
        {
            get
            {
                foreach (int i in m_BranchTargetOffsets)
                    yield return i;
            }
        }

        public override string ToString()
        {
            string str = "";
            foreach (int i in m_BranchTargetOffsets)
                str += "IL_" + i.ToString("X4") + " ";

            return base.ToString() + " " + str;
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1}", IndentString(indent), this.ToString());
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

            for(int i=0; i<m_BranchTargetOffsets.Count; i++) // Add all conditional branches 
            {
                HighLevel.Node Argument = new HighLevel.EqualsNode(Context.ReadStackLocationNode(Context.StackPointer), new HighLevel.IntegerConstantNode(i));
                List.Add(new HighLevel.ConditionalBranchInstruction(Argument, Context.GetBlock(m_BranchTargetOffsets[i])));
            }

            return List;
        }
    }
}
