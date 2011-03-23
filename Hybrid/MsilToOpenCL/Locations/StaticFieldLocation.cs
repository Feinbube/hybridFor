using System;
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
