/*    
*    CilLoadArgumentInstruction.cs
*
﻿*    Copyright (C) 2012 Jan-Arne Sobania, Frank Feinbube, Ralf Diestelkämper
*
*    This library is free software: you can redistribute it and/or modify
*    it under the terms of the GNU Lesser General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public License
*    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*    jan-arne [dot] sobania [at] gmx [dot] net
*    Frank [at] Feinbube [dot] de
*    ralf [dot] diestelkaemper [at] hotmail [dot] com
*
*/


﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Hybrid.MsilToOpenCL.Instructions
{
    public class CilLoadArgumentInstruction : CilInstruction
    {
        private int m_Index;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase)
        {
            int Index;
            if (Opcode == OpCodes.Ldarg_S)
            {
                Index = IL[Offset + Opcode.Size + 0];
            }
            else if (Opcode == OpCodes.Ldarg)
            {
                Index = ReadUInt16(IL, Offset + Opcode.Size);
            }
            else
            {
                Index = -1;
            }

            return new CilLoadArgumentInstruction(Opcode, Offset, Index);
        }

        private CilLoadArgumentInstruction(OpCode Opcode, int Offset, int Index)
            : base(Opcode, Offset)
        {
            if (Opcode == OpCodes.Ldarg_0)
            {
                m_Index = 0;
            }
            else if (Opcode == OpCodes.Ldarg_1)
            {
                m_Index = 1;
            }
            else if (Opcode == OpCodes.Ldarg_2)
            {
                m_Index = 2;
            }
            else if (Opcode == OpCodes.Ldarg_3)
            {
                m_Index = 3;
            }
            else if (Opcode == OpCodes.Ldarg || Opcode == OpCodes.Ldarg_S)
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
            return base.ToString() + ((Opcode == OpCodes.Ldarg || Opcode == OpCodes.Ldarg_S) ? " " + m_Index : string.Empty);
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack)
        {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), "argument_" + m_Index.ToString());
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context)
        {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1), Context.ArgumentNode(m_Index)));
            return List;
        }
    }
}
