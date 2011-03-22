﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilStoreLocalInstruction : CilInstruction
    {
        private int m_Index;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int Index;
            if (Opcode == OpCodes.Stloc_S)
            {
                Index = ReadUInt8(IL, Offset + Opcode.Size);
            }
            else if (Opcode == OpCodes.Stloc)
            {
                Index = ReadUInt16(IL, Offset + Opcode.Size);
            }
            else
            {
                Index = -1;
            }

            return new CilStoreLocalInstruction(Opcode, Offset, Index);
        }

        private CilStoreLocalInstruction(OpCode Opcode, int Offset, int Index)
            : base(Opcode, Offset)
        {
            if (Opcode == OpCodes.Stloc_0)
            {
                m_Index = 0;
            }
            else if (Opcode == OpCodes.Stloc_1)
            {
                m_Index = 1;
            }
            else if (Opcode == OpCodes.Stloc_2)
            {
                m_Index = 2;
            }
            else if (Opcode == OpCodes.Stloc_3)
            {
                m_Index = 3;
            }
            else if (Opcode == OpCodes.Stloc || Opcode == OpCodes.Stloc_S)
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
            return base.ToString() + ((Opcode == OpCodes.Stloc || Opcode == OpCodes.Stloc_S) ? " " + m_Index : string.Empty);
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = {2};", IndentString(indent), "local_" + m_Index.ToString(), StackName(CurStack));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.AssignmentInstruction(Context.LocalVariableNode(m_Index), Context.ReadStackLocationNode(Context.StackPointer)));
            return List;
        }
    }
}
