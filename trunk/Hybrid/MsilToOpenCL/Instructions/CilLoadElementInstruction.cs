using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilLoadElementInstruction : CilInstruction
    {
        private Type m_ArrayType;

        public static CilInstruction CreateWithType(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int Token = ReadInt32(IL, Offset + Opcode.Size);
            Type ElementType = ParentMethodBase.Module.ResolveType(Token);
            Type ArrayType = Array.CreateInstance(ElementType, 1).GetType();
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, ArrayType);
        }

        public static CilInstruction Create_I(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(IntPtr[]));
        }

        public static CilInstruction Create_I1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(sbyte[]));
        }

        public static CilInstruction Create_I2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(short[]));
        }

        public static CilInstruction Create_I4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(int[]));
        }

        public static CilInstruction Create_I8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(long[]));
        }

        public static CilInstruction Create_U1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(byte[]));
        }

        public static CilInstruction Create_U2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(ushort[]));
        }

        public static CilInstruction Create_U4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(uint[]));
        }

        public static CilInstruction Create_R4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(float[]));
        }

        public static CilInstruction Create_R8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(double[]));
        }

        public static CilInstruction Create_Ref(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(object[]));
        }

        private static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase, Type ArrayType)
        {
            return new CilLoadElementInstruction(Opcode, Offset, ArrayType);
        }

        private CilLoadElementInstruction(OpCode Opcode, int Offset, Type ArrayType)
            : base(Opcode, Offset)
        {
            System.Diagnostics.Debug.Assert(ArrayType.IsArray);
            m_ArrayType = ArrayType;
        }

        public Type ArrayType
        {
            get
            {
                return m_ArrayType;
            }
        }

        public override string ToString()
        {
            return base.ToString() + " " + m_ArrayType.GetElementType();
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = ({2})[{3}];", IndentString(indent), StackName(CurStack - 1), StackName(CurStack - 1), StackName(CurStack));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

            HighLevel.ArrayAccessNode Argument = new HighLevel.ArrayAccessNode(ArrayType);
            Argument.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer - 1));
            Argument.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer));

            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer - 1), Argument));
            return List;
        }
    }
}
