using System;
using System.Collections.Generic;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
	public enum LocationType {
		CilStack,
		Argument,
		LocalVariable,
		StaticField
	}

	[Flags]
	public enum LocationFlags {
		Read = 1,
		IndirectRead = 2,
		Write = 4,
		IndirectWrite = 8
	}

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

	public class StackLocation : Location {
		private int m_Index;

		public StackLocation(int Index)
			: this(Index, null) {
		}

		public StackLocation(int Index, Type DataType)
			: base(LocationType.CilStack, "__cil_stack_" + Index.ToString(), "__cil_stack_" + Index.ToString(), DataType) {
			m_Index = Index;
		}

		protected StackLocation(StackLocation ex)
			: base(ex) {
			m_Index = ex.m_Index;
		}

		public int Index { get { return m_Index; } }

		public override string ToString() {
			return "[stack(" + Index.ToString() + ")] " + (DataType == null ? "??? " : DataType.ToString() + " ") + Name;
		}

		public override int GetHashCode() {
			return m_Index;
		}

		protected override bool InnerEquals(Location obj) {
			return ((StackLocation)obj).m_Index == m_Index;
		}

		internal override int CompareToLocation(Location Other) {
			if (object.ReferenceEquals(Other, null) || (Other.LocationType != this.LocationType)) {
				throw new ArgumentException("Other");
			}

			return m_Index.CompareTo(((StackLocation)Other).m_Index);
		}

		#region ICloneable Members

		public override object Clone() {
			return new StackLocation(this);
		}

		#endregion
	}

	public class ArgumentLocation : Location {
		private int m_Index;
		private bool m_FromIL;

		public ArgumentLocation(int Index, string Name, Type DataType, bool FromIL)
			: base(LocationType.Argument, Name, Name, DataType) {
			m_Index = Index;
			m_FromIL = FromIL;
		}

		protected ArgumentLocation(ArgumentLocation ex)
			: base(ex) {
			m_Index = ex.m_Index;
		}

		public int Index { get { return m_Index; } set { m_Index = value; } }
		public bool FromIL { get { return m_FromIL; } }

		public override string ToString() {
			return "[param(" + Index.ToString() + ")] " + (DataType == null ? "??? " : DataType.ToString() + " ") + Name;
		}

		public override int GetHashCode() {
			return m_Index.GetHashCode();
		}

		protected override bool InnerEquals(Location obj) {
			return ((ArgumentLocation)obj).m_Index == m_Index;
		}

		internal override int CompareToLocation(Location Other) {
			if (object.ReferenceEquals(Other, null) || (Other.LocationType != this.LocationType)) {
				throw new ArgumentException("Other");
			}

			return m_Index.CompareTo(((ArgumentLocation)Other).m_Index);
		}

		#region ICloneable members

		public override object Clone() {
			return new ArgumentLocation(this);
		}

		#endregion
	}

	public class LocalVariableLocation : Location {
		private int m_Index;

		public LocalVariableLocation(int Index, string Name, Type DataType)
			: base(LocationType.LocalVariable, "local_" + Index.ToString(), Name, DataType) {
			m_Index = Index;
		}

		protected LocalVariableLocation(LocalVariableLocation ex)
			: base(ex) {
			m_Index = ex.m_Index;
		}

		public int Index { get { return m_Index; } }

		public override string ToString() {
			return "[local] " + (DataType == null ? "??? " : DataType.ToString() + " ") + Name;
		}

		public override int GetHashCode() {
			return m_Index.GetHashCode();
		}

		protected override bool InnerEquals(Location obj) {
			return ((LocalVariableLocation)obj).m_Index == m_Index;
		}

		internal override int CompareToLocation(Location Other) {
			if (object.ReferenceEquals(Other, null) || (Other.LocationType != this.LocationType)) {
				throw new ArgumentException("Other");
			}

			return m_Index.CompareTo(((LocalVariableLocation)Other).m_Index);
		}

		#region ICloneable members

		public override object Clone() {
			return new LocalVariableLocation(this);
		}

		#endregion
	}

	public class StaticFieldLocation : Location {
		System.Reflection.FieldInfo m_FieldInfo;

		public StaticFieldLocation(System.Reflection.FieldInfo FieldInfo)
			: base(LocationType.StaticField, FieldInfo.Name, FieldInfo.Name, FieldInfo.FieldType) {
			m_FieldInfo = FieldInfo;
		}

		protected StaticFieldLocation(StaticFieldLocation ex)
			: base(ex) {
			m_FieldInfo = ex.m_FieldInfo;
		}

		public System.Reflection.FieldInfo FieldInfo { get { return m_FieldInfo; } }

		public override string ToString() {
			return "[field] " + FieldInfo.ToString();
		}

		public override int GetHashCode() {
			return m_FieldInfo.GetHashCode();
		}

		protected override bool InnerEquals(Location obj) {
			return object.Equals(((StaticFieldLocation)obj).m_FieldInfo, m_FieldInfo);
		}

		internal override int CompareToLocation(Location Other) {
			throw new NotImplementedException();
		}

		#region ICloneable members

		public override object Clone() {
			return new StaticFieldLocation(this);
		}

		#endregion
	}
}
