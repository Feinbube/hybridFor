using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilUnaryNumericInstruction : CilInstruction
    {
        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return new CilUnaryNumericInstruction(Opcode, Offset);
        }

        private CilUnaryNumericInstruction(OpCode Opcode, int Offset)
            : base(Opcode, Offset)
        {
            if (!(Opcode == OpCodes.Neg))
            {
                throw new ArgumentException("Opcode");
            }
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = negate({2})", IndentString(indent), StackName(CurStack), StackName(CurStack));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.AssignmentInstruction(Context.ReadStackLocationNode(Context.StackPointer), new HighLevel.NegNode(Context.ReadStackLocationNode(Context.StackPointer))));
            return List;
        }
    }
}
