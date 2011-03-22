﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilConvertInstruction : CilInstruction
    {
        private Type m_Type;

        public static CilInstruction Create_I(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(IntPtr));
        }

        public static CilInstruction Create_I1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(sbyte));
        }

        public static CilInstruction Create_I2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(short));
        }

        public static CilInstruction Create_I4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(int));
        }

        public static CilInstruction Create_I8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(long));
        }

        public static CilInstruction Create_U(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(UIntPtr));
        }

        public static CilInstruction Create_U1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(byte));
        }

        public static CilInstruction Create_U2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(ushort));
        }

        public static CilInstruction Create_U4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(uint));
        }

        public static CilInstruction Create_R4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(float));
        }

        public static CilInstruction Create_R8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(double));
        }

        private static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase, Type Type)
        {
            return new CilConvertInstruction(Opcode, Offset, Type);
        }

        private CilConvertInstruction(OpCode Opcode, int Offset, Type Type)
            : base(Opcode, Offset)
        {
            m_Type = Type;
        }

        public Type Type
        {
            get
            {
                return m_Type;
            }
        }

        public override string ToString()
        {
            return base.ToString() + " " + m_Type.ToString();
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = ({2})[{3}];", IndentString(indent), StackName(CurStack), m_Type.ToString(), StackName(CurStack));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer), new HighLevel.CastNode(Context.ReadStackLocationNode(Context.StackPointer), Type)));
            return List;
        }
    }
}
