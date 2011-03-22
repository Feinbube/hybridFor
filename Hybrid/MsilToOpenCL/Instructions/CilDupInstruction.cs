using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Reflection.Emit;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilDupInstruction : CilInstruction
    {
        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return new CilDupInstruction(Opcode, Offset);
        }

        private CilDupInstruction(OpCode Opcode, int Offset)
            : base(Opcode, Offset)
        {
            if (!(Opcode == OpCodes.Dup))
            {
                throw new ArgumentException("Opcode");
            }
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = {2}", IndentString(indent), StackName(CurStack + 1), StackName(CurStack));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

            HighLevel.LocationNode Argument = Context.ReadStackLocationNode(Context.StackPointer);
            Context.DefineStackLocationNode(Context.StackPointer, Argument.Location.DataType);

            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1), Argument));
            return List;
        }
    }
}
