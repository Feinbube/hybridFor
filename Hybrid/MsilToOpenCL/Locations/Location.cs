using System;
using System.Collections.Generic;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
	public abstract class Location : ICloneable {
		private LocationType m_LocationType;
		private LocationFlags m_Flags;
		private string m_Name;
		private string m_OriginalName;
		private Type m_DataType;

		public Location(LocationType LocationType, string Name, string OriginalName, Type DataType) {
			m_LocationType = LocationType;
			m_Name = Name;
			m_OriginalName = OriginalName;
			m_DataType = DataType;
		}

		protected Location(Location ex) {
			m_LocationType = ex.m_LocationType;
			m_Name = ex.m_Name;
			m_DataType = ex.m_DataType;
		}

		public LocationFlags Flags { get { return m_Flags; } set { m_Flags = value; } }
		public LocationType LocationType { get { return m_LocationType; } }
		public string Name { get { return m_Name; } set { m_Name = value; } }
		public string OriginalName { get { return m_OriginalName; } set { m_OriginalName = value; } }
		public Type DataType { get { return m_DataType; } set { m_DataType = value; } }

		public override bool Equals(object obj) {
			if (object.ReferenceEquals(obj,null)) {
				return false;
			} else if (object.ReferenceEquals(obj,this)) {
				return true;
			} else if (obj.GetType() != this.GetType()) {
				return false;
			}

			return InnerEquals((Location)obj);
		}

		public override int GetHashCode() {
			return (m_Name != null ? m_Name.GetHashCode() : 0) + (m_DataType != null ? m_DataType.GetHashCode() : 0);
		}

		protected virtual bool InnerEquals(Location obj) {
			return (obj.LocationType == m_LocationType && obj.m_Name == m_Name && obj.m_DataType == m_DataType);
		}

		#region ICloneable Members
		public abstract object Clone();
		#endregion

		internal abstract int CompareToLocation(Location Other);
	}
}
