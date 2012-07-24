/*    
*    StaticFieldLocation.cs
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
    public class StaticFieldLocation : Location
    {
        System.Reflection.FieldInfo m_FieldInfo;

        public StaticFieldLocation(System.Reflection.FieldInfo FieldInfo)
            : base(LocationType.StaticField, FieldInfo.Name, FieldInfo.Name, FieldInfo.FieldType)
        {
            m_FieldInfo = FieldInfo;
        }

        protected StaticFieldLocation(StaticFieldLocation ex)
            : base(ex)
        {
            m_FieldInfo = ex.m_FieldInfo;
        }

        public System.Reflection.FieldInfo FieldInfo { get { return m_FieldInfo; } }

        public override string ToString()
        {
            return "[field] " + FieldInfo.ToString();
        }

        public override int GetHashCode()
        {
            return m_FieldInfo.GetHashCode();
        }

        protected override bool InnerEquals(Location obj)
        {
            return object.Equals(((StaticFieldLocation)obj).m_FieldInfo, m_FieldInfo);
        }

        internal override int CompareToLocation(Location Other)
        {
            throw new NotImplementedException();
        }

        #region ICloneable members

        public override object Clone()
        {
            return new StaticFieldLocation(this);
        }

        #endregion
    }
}
