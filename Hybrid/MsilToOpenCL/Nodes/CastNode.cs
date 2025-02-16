/*    
*    CastNode.cs
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

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class CastNode : Node
    {
        private System.Type m_Type;

        public CastNode(Node Argument, Type Type)
            : base(NodeType.Cast, Type, false)
        {
            m_Type = Type;
            SubNodes.Add(Argument);
        }

        public Type Type { get { return m_Type; } }

        public override string ToString()
        {
            return "((" + OpenCLInterop.GetOpenClType(this.HlGraph, m_Type) + ")(" + (SubNodes.Count == 0 ? "???" : SubNodes[0].ToString()) + "))";
        }
    }
}
