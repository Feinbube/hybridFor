using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class LocalVariableLocation : Location
    {
        private int m_Index;

        public LocalVariableLocation(int Index, string Name, Type DataType)
            : base(LocationType.LocalVariable, Name, Name, DataType)
        {
            m_Index = Index;
        }

        protected LocalVariableLocation(LocalVariableLocation ex)
            : base(ex)
        {
            m_Index = ex.m_Index;
        }

        public int Index { get { return m_Index; } }

        public override string ToString()
        {
            return "[local] " + (DataType == null ? "??? " : DataType.ToString() + " ") + Name;
        }

        public override int GetHashCode()
        {
            return m_Index.GetHashCode();
        }

        protected override bool InnerEquals(Location obj)
        {
            return ((LocalVariableLocation)obj).m_Index == m_Index;
        }

        internal override int CompareToLocation(Location Other)
        {
            if (object.ReferenceEquals(Other, null) || (Other.LocationType != this.LocationType))
            {
                throw new ArgumentException("Other");
            }

            return m_Index.CompareTo(((LocalVariableLocation)Other).m_Index);
        }

        #region ICloneable members

        public override object Clone()
        {
            return new LocalVariableLocation(this);
        }

        #endregion
    }
}
