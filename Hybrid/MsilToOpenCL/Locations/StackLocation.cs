using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class StackLocation : Location
    {
        private int m_Index;

        public StackLocation(int Index)
            : this(Index, null)
        {
        }

        public StackLocation(int Index, Type DataType)
            : base(LocationType.CilStack, "__cil_stack_" + Index.ToString(), "__cil_stack_" + Index.ToString(), DataType)
        {
            m_Index = Index;
        }

        protected StackLocation(StackLocation ex)
            : base(ex)
        {
            m_Index = ex.m_Index;
        }

        public int Index { get { return m_Index; } }

        public override string ToString()
        {
            return "[stack(" + Index.ToString() + ")] " + (DataType == null ? "??? " : DataType.ToString() + " ") + Name;
        }

        public override int GetHashCode()
        {
            return m_Index;
        }

        protected override bool InnerEquals(Location obj)
        {
            return base.InnerEquals(obj) && ((StackLocation)obj).m_Index == m_Index;
        }

        internal override int CompareToLocation(Location Other)
        {
            if (object.ReferenceEquals(Other, null) || (Other.LocationType != this.LocationType))
            {
                throw new ArgumentException("Other");
            }

            return m_Index.CompareTo(((StackLocation)Other).m_Index);
        }

        #region ICloneable Members

        public override object Clone()
        {
            return new StackLocation(this);
        }

        #endregion
    }
}
