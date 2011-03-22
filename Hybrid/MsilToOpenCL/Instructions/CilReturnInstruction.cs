using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilReturnInstruction : CilInstruction
    {
        private Type m_ReturnType;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            Type ReturnType = typeof(void);
            if (!ParentMethodBase.IsConstructor)
            {
                System.Diagnostics.Debug.Assert(ParentMethodBase is MethodInfo);
                ReturnType = ((MethodInfo)ParentMethodBase).ReturnType;
            }

            return new CilReturnInstruction(Opcode, Offset, ReturnType);
        }

        private CilReturnInstruction(OpCode Opcode, int Offset, Type ReturnType)
            : base(Opcode, Offset)
        {
            if (Opcode != OpCodes.Ret)
            {
                throw new ArgumentException();
            }
            m_ReturnType = ReturnType;
        }

        public override bool CanFallThrough
        {
            get
            {
                return false;
            }
        }

        public override int StackConsumeCount
        {
            get
            {
                System.Diagnostics.Debug.Assert(Opcode == OpCodes.Ret);
                System.Diagnostics.Debug.Assert(Opcode.StackBehaviourPop == StackBehaviour.Varpop);

                return (m_ReturnType == typeof(void)) ? 0 : 1;
            }
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);

            string ReturnArgument;
            if (StackConsumeCount == 0)
            {
                ReturnArgument = string.Empty;
            }
            else
            {
                ReturnArgument = " (" + StackName(CurStack) + ")";
            }
            writer.WriteLine("{0}return{1};", IndentString(indent), ReturnArgument);
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            HighLevel.Node Argument;

            if (StackConsumeCount == 0)
            {
                Argument = null;
            }
            else
            {
                Argument = Context.ReadStackLocationNode(Context.StackPointer);
            }

            List.Add(new HighLevel.ReturnInstruction(Argument));
            return List;
        }
    }
}
