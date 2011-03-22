using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilLoadR4ConstantInstruction : CilInstruction
    {
        private float m_Constant;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return new CilLoadR4ConstantInstruction(Opcode, Offset, ReadFloat(IL, Offset + Opcode.Size));
        }

        private CilLoadR4ConstantInstruction(OpCode Opcode, int Offset, float Constant)
            : base(Opcode, Offset)
        {
            if (Opcode != OpCodes.Ldc_R4)
            {
                throw new ArgumentException();
            }
            m_Constant = Constant;
        }

        public float Constant
        {
            get
            {
                return m_Constant;
            }
        }

        public override string ToString()
        {
            return base.ToString() + " " + m_Constant.ToString("F1") + "f";
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), m_Constant);
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1, typeof(float)), new HighLevel.FloatConstantNode(m_Constant)));
            return List;
        }
    }
}
