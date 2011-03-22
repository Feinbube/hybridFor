using System;
using System.Collections.Generic;
using System.Text;

namespace Ever.HighLevel {
	public enum NodeType {
		StringConstant,
		IntegerConstant,
		FloatConstant,
		DoubleConstant,
		Location,

		InstanceField,

		Call,
		ArrayAccess,
		Cast,

		Neg,
		LogicalNot,
		AddressOf,
        Deref,

		Equals,
		NotEquals,
		Less,
		LessEquals,
		Greater,
		GreaterEquals,

		Add,
		Sub,
		Mul,
		Div,
		Mod,
	}

	public abstract class Node {
		private NodeType m_NodeType;
		private Type m_DataType;
		private List<Node> m_SubNodes;
		private HlGraph m_HlGraph;

		protected Node(NodeType NodeType, Type DataType, bool IsLeaf) {
			m_NodeType = NodeType;
			m_DataType = DataType;
			m_SubNodes = new List<Node>();
		}

		public HlGraph HlGraph { get { return m_HlGraph; } set { m_HlGraph = value; } }
		public NodeType NodeType { get { return m_NodeType; } }
		public virtual Type DataType { get { return m_DataType; } set { m_DataType = value; } }
		public List<Node> SubNodes { get { return m_SubNodes; } }
	}

	public class StringConstantNode : Node {
		private string m_Value;

		public StringConstantNode(string Value)
			: base(NodeType.StringConstant, typeof(string), true) {
			m_Value = Value;
		}

		public string Value {
			get {
				return m_Value;
			}
		}

		public override string ToString() {
			return "\"" + m_Value + "\"";
		}
	}

	public class IntegerConstantNode : Node {
		private ulong m_Value;

		public IntegerConstantNode(int Value)
			: base(NodeType.IntegerConstant, typeof(int), true) {
			m_Value = (ulong)(long)Value;
		}

		public IntegerConstantNode(uint Value)
			: base(NodeType.IntegerConstant, typeof(uint), true) {
			m_Value = (ulong)Value;
		}

		public IntegerConstantNode(long Value)
			: base(NodeType.IntegerConstant, typeof(long), true) {
			m_Value = (ulong)Value;
		}

		public IntegerConstantNode(ulong Value)
			: base(NodeType.IntegerConstant, typeof(ulong), true) {
			m_Value = Value;
		}

		public object Value {
			get {
				return m_Value;
			}
		}

		public override string ToString() {
			return m_Value.ToString();
		}
	}

	public class FloatConstantNode : Node {
		private float m_Value;

		public FloatConstantNode(float Value)
			: base(NodeType.FloatConstant, typeof(float), true) {
			m_Value = Value;
		}

		public float Value {
			get {
				return m_Value;
			}
		}

		public override string ToString() {
			return "((float)" + m_Value.ToString("r", System.Globalization.NumberFormatInfo.InvariantInfo) + ")";
		}
	}

	public class DoubleConstantNode : Node {
		private double m_Value;

		public DoubleConstantNode(double Value)
			: base(NodeType.DoubleConstant, typeof(double), true) {
			m_Value = Value;
		}

		public double Value {
			get {
				return m_Value;
			}
		}

		public override string ToString() {
			return "((double)" + m_Value.ToString("r", System.Globalization.NumberFormatInfo.InvariantInfo) + ")";
		}
	}

	public class LocationNode : Node {
		private Location m_Location;

		public LocationNode(Location Location)
			: base(NodeType.Location, Location.DataType, true) {
			m_Location = Location;
		}

		public Location Location {
			get {
				return m_Location;
			}
			set {
				m_Location = value;
			}
		}

		public override Type DataType {
			get {
				return base.DataType;
			}
			set {
				base.DataType = value;
				if (m_Location != null && m_Location.DataType == null) {
					m_Location.DataType = value;
				}
			}
		}

		public override string ToString() {
			return m_Location.Name;
		}
	}

	public class InstanceFieldNode : Node {
		private System.Reflection.FieldInfo m_FieldInfo;

		public InstanceFieldNode(Node InstanceNode, System.Reflection.FieldInfo FieldInfo)
			: base(NodeType.InstanceField, FieldInfo.FieldType, false) {
			SubNodes.Add(InstanceNode);
			m_FieldInfo = FieldInfo;
		}

		public System.Reflection.FieldInfo FieldInfo { get { return m_FieldInfo; } }

		public override string ToString() {
			if (SubNodes.Count == 0) {
				return "(???).__field_ref[\"" + (FieldInfo == null ? "???" : FieldInfo.ToString()) + "\"]";
			} else if (SubNodes.Count == 1) {
				return SubNodes[0].ToString() + "." + ((FieldInfo == null) ? "???" : FieldInfo.Name);
			} else {
				return "(??? <too many childs for instance field access node> ???)";
			}
		}
	}

	public class CallNode : Node {
		private System.Reflection.MethodInfo m_MethodInfo;

		public CallNode(System.Reflection.MethodInfo MethodInfo)
			: base(NodeType.Call, MethodInfo.ReturnType, false) {
			m_MethodInfo = MethodInfo;
		}

		public CallNode(System.Reflection.MethodInfo MethodInfo, params Node[] Arguments)
			: this(MethodInfo) {
			SubNodes.AddRange(Arguments);
		}

		public System.Reflection.MethodInfo MethodInfo { get { return m_MethodInfo; } }

		public override string ToString() {
			StringBuilder String = new StringBuilder();
			int i = 0;

			if ((MethodInfo.CallingConvention & System.Reflection.CallingConventions.HasThis) != 0) {
				if (SubNodes.Count == 0) {
					String.Append("(???).");
				} else {
					String.Append(SubNodes[0].ToString());
					String.Append(".");
				}
				i++;
			}

			string Name = object.ReferenceEquals(HlGraph, null) ? OpenClAliasAttribute.Get(MethodInfo) : HlGraph.GetOpenClFunctionName(MethodInfo);
			if (Name == null) {
				Name = MethodInfo.Name;
			}
			String.Append(Name);
			String.Append("(");

			bool IsFirst = true;
			for (; i < SubNodes.Count; i++) {
				if (IsFirst) {
					IsFirst = false;
				} else {
					String.Append(", ");
				}
				String.Append(SubNodes[i].ToString());
			}

			String.Append(")");

			return String.ToString();
		}
	}

	public class CastNode : Node {
		private System.Type m_Type;

		public CastNode(Node Argument, Type Type)
			: base(NodeType.Cast, Type, false) {
			m_Type = Type;
			SubNodes.Add(Argument);
		}

		public Type Type { get { return m_Type; } }

		public override string ToString() {
			return "((" + Ever.Parallel.GetOpenClType(m_Type) + ")(" + (SubNodes.Count == 0 ? "???" : SubNodes[0].ToString()) + "))";
		}
	}

	public class ArrayAccessNode : Node {
		private System.Type m_ArrayType;

		public ArrayAccessNode(Type ArrayType)
			: base(NodeType.ArrayAccess, ArrayType.GetElementType(), false) {
			if (!ArrayType.IsArray) {
				throw new ArgumentException("ArrayAccessNode requires an array type.");
			}

			m_ArrayType = ArrayType;
		}

		public Type ArrayType { get { return m_ArrayType; } }

		public override string ToString() {
			if (SubNodes.Count == 0) {
				return "(???)[ ??? ]";
			}

			StringBuilder String = new StringBuilder();
			String.Append(SubNodes[0].ToString());
			String.Append("[");

			if (SubNodes.Count == 1) {
				String.Append(" ??? ]");
			} else {
				for (int i = 1; i < SubNodes.Count; i++) {
					if (i > 1) {
						String.Append(", ");
					}
					String.Append(SubNodes[i].ToString());
				}
				String.Append("]");
			}

			return String.ToString();
		}

		internal void FlattenArrayType() {
			if (ArrayType.GetArrayRank() > 1) {
				m_ArrayType = System.Array.CreateInstance(ArrayType.GetElementType(), 1).GetType();
			}
		}
	}

	public abstract class UnaryOperatorNode : Node {
		public UnaryOperatorNode(NodeType NodeType, Node Argument)
			: base(NodeType, null, false) {
			SubNodes.Add(Argument);

			this.DataType = GetResultType(Argument);
		}

		public override string ToString() {
			if (SubNodes.Count == 0) {
				return Symbol + "(???)";
			} else if (SubNodes.Count == 1) {
				return Symbol + SubNodes[0].ToString();
			} else {
				return "(??? <too many childs for unary operator node> ???)";
			}
		}

		public virtual Type GetResultType(Node Argument) {
			if (Argument == null || Argument.DataType == null) {
				return null;
			}

			Type Type = Argument.DataType;

			if (Type == typeof(int) || Type == typeof(long) || Type == typeof(IntPtr) || Type == typeof(float) || Type == typeof(double)) {
				return Type;
			}

			// TODO
			return null;
		}

		public abstract string Symbol { get;}
	}

	public class NegNode : UnaryOperatorNode {
		public NegNode(Node Argument)
			: base(NodeType.Neg, Argument) {
		}

		public override string Symbol { get { return "-"; } }
	}

	public class LogicalNotNode : UnaryOperatorNode {
		public LogicalNotNode(Node Argument)
			: base(NodeType.LogicalNot, Argument) {
		}

		public override string Symbol { get { return "!"; } }
	}

	public class AddressOfNode : UnaryOperatorNode {
		public AddressOfNode(Node Argument)
			: base(NodeType.AddressOf, Argument) {
		}

		public override string Symbol { get { return "&"; } }

		public override Type GetResultType(Node Argument) {
			if (Argument == null || Argument.DataType == null) {
				return null;
			}

			Type Type = Argument.DataType;

			if (!Type.IsPointer && !Type.IsByRef) {
				return Type.Assembly.GetType(Type.FullName + "*", true);
			}

			// TODO
			return null;
		}
	}

    public class DerefNode : UnaryOperatorNode {
        public DerefNode(Node Argument)
            : base(NodeType.Deref, Argument) {
        }

        public override string Symbol { get { return "*"; } }

        public override Type GetResultType(Node Argument) {
            if (Argument == null || Argument.DataType == null) {
                return null;
            }

            Type Type = Argument.DataType;

            if (Type.IsPointer || Type.IsByRef) {
                return Type.GetElementType();
            }

            // TODO
            return null;
        }
    }

	public abstract class BinaryOperatorNode : Node {
		public BinaryOperatorNode(NodeType NodeType, Node Left, Node Right)
			: base(NodeType, null, false) {
			SubNodes.Add(Left);
			SubNodes.Add(Right);

			this.DataType = GetResultType(Left, Right);
		}

		public override string ToString() {
			if (SubNodes.Count == 0) {
				return "(??? " + Symbol + " ???)";
			} else if (SubNodes.Count == 1) {
				return "(" + SubNodes[0].ToString() + " " + Symbol + " ???)";
			} else if (SubNodes.Count == 2) {
				return "(" + SubNodes[0].ToString() + " " + Symbol + " " + SubNodes[1].ToString() + ")";
			} else {
				return "(??? <too many childs for binary operator node> ???)";
			}
		}

		public virtual Type GetResultType(Node Left, Node Right) {
			if (Left == null || Left.DataType == null || Right == null || Right.DataType == null) {
				return null;
			}

			Type LeftType = Left.DataType, RightType = Right.DataType;

			if (LeftType == typeof(int)) {
				if (RightType == typeof(int) || RightType == typeof(IntPtr) || RightType == typeof(uint)) {
					return RightType;
				}
			} else if (LeftType == typeof(uint)) {
				if (RightType == typeof(int) || RightType == typeof(uint)) {
					return LeftType;
				} else if (RightType == typeof(IntPtr)) {
					return RightType;
				}
			} else if (LeftType == typeof(long)) {
				if (RightType == LeftType) {
					return RightType;
				}
			} else if (LeftType == typeof(IntPtr)) {
				if (RightType == typeof(int) || RightType == typeof(IntPtr)) {
					return RightType;
				}
			} else if (LeftType == typeof(float) || LeftType == typeof(double)) {
				if (RightType == typeof(float) || RightType == typeof(double)) {
					return (LeftType == typeof(double) || RightType == typeof(double)) ? typeof(double) : typeof(float);
				}
			}

			// TODO
			return null;
		}

		public abstract string Symbol {get;}
	}

	public class AddNode : BinaryOperatorNode {
		public AddNode(Node Left, Node Right)
			: base(NodeType.Add, Left, Right) {
		}

		public override string Symbol { get { return "+"; } }
	}

	public class SubNode : BinaryOperatorNode {
		public SubNode(Node Left, Node Right)
			: base(NodeType.Sub, Left, Right) {
		}

		public override string Symbol { get { return "-"; } }
	}

	public class MulNode : BinaryOperatorNode {
		public MulNode(Node Left, Node Right)
			: base(NodeType.Mul, Left, Right) {
		}

		public override string Symbol { get { return "*"; } }
	}

	public class DivNode : BinaryOperatorNode {
		public DivNode(Node Left, Node Right)
			: base(NodeType.Div, Left, Right) {
		}

		public override string Symbol { get { return "/"; } }
	}

	public class ModNode : BinaryOperatorNode {
		public ModNode(Node Left, Node Right)
			: base(NodeType.Mod, Left, Right) {
		}

		public override string Symbol { get { return "%"; } }
	}

	public abstract class BinaryComparisonOperatorNode : BinaryOperatorNode {
		public BinaryComparisonOperatorNode(NodeType NodeType, Node Left, Node Right)
			: base(NodeType, Left, Right) {
		}

		public override Type GetResultType(Node Left, Node Right) {
			return typeof(int);
		}
	}

	public class EqualsNode : BinaryComparisonOperatorNode {
		public EqualsNode(Node Left, Node Right)
			: base(NodeType.Equals, Left, Right) {
		}

		public override string Symbol {
			get { return "=="; }
		}
	}

	public class NotEqualsNode : BinaryComparisonOperatorNode {
		public NotEqualsNode(Node Left, Node Right)
			: base(NodeType.NotEquals, Left, Right) {
		}

		public override string Symbol {
			get { return "!="; }
		}
	}

	public class LessEqualsNode : BinaryComparisonOperatorNode {
		public LessEqualsNode(Node Left, Node Right)
			: base(NodeType.LessEquals, Left, Right) {
		}

		public override string Symbol {
			get { return "<="; }
		}
	}

	public class LessNode : BinaryComparisonOperatorNode {
		public LessNode(Node Left, Node Right)
			: base(NodeType.Less, Left, Right) {
		}

		public override string Symbol {
			get { return "<"; }
		}
	}

	public class GreaterEqualsNode : BinaryComparisonOperatorNode {
		public GreaterEqualsNode(Node Left, Node Right)
			: base(NodeType.GreaterEquals, Left, Right) {
		}

		public override string Symbol {
			get { return ">="; }
		}
	}

	public class GreaterNode : BinaryComparisonOperatorNode {
		public GreaterNode(Node Left, Node Right)
			: base(NodeType.Greater, Left, Right) {
		}

		public override string Symbol {
			get { return ">"; }
		}
	}
}
