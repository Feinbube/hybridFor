using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class ArgumentLocation : Location
    {
        private int m_Index;
        private bool m_FromIL;

        public ArgumentLocation(int Index, string Name, Type DataType, bool FromIL)
            : base(LocationType.Argument, Name, Name, DataType)
        {
            m_Index = Index;
            m_FromIL = FromIL;
        }

        protected ArgumentLocation(ArgumentLocation ex)
            : base(ex)
        {
            m_Index = ex.m_Index;
        }

        public int Index { get { return m_Index; } set { m_Index = value; } }
        public bool FromIL { get { return m_FromIL; } }

        public override string ToString()
        {
            return "[param(" + Index.ToString() + ")] " + (DataType == null ? "??? " : DataType.ToString() + " ") + Name;
        }

        public override int GetHashCode()
        {
            return m_Index.GetHashCode();
        }

        protected override bool InnerEquals(Location obj)
        {
            return ((ArgumentLocation)obj).m_Index == m_Index;
        }

        internal override int CompareToLocation(Location Other)
        {
            if (object.ReferenceEquals(Other, null) || (Other.LocationType != this.LocationType))
            {
                throw new ArgumentException("Other");
            }

            return m_Index.CompareTo(((ArgumentLocation)Other).m_Index);
        }

        #region ICloneable members

        public override object Clone()
        {
            return new ArgumentLocation(this);
        }

        #endregion
    }
}
