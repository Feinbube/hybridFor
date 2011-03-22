using System;
using System.Collections.Generic;
using System.Text;
using System.Reflection.Emit;
using System.Reflection;

namespace Ever {
	public abstract class CilInstruction {
		private int m_Offset;
		private OpCode m_Opcode;

		public CilInstruction(OpCode Opcode, int Offset) {
			m_Opcode = Opcode;
			m_Offset = Offset;
		}

		public OpCode Opcode {
			get {
				return m_Opcode;
			}
		}

		public int Offset {
			get {
				return m_Offset;
			}
		}

		public override string ToString() {
			return m_Offset.ToString("X8") + "    " + m_Opcode.ToString();
		}

		public virtual bool CanFallThrough {
			get {
				return true;
			}
		}

		public virtual IEnumerable<int> BranchTargetOffsets {
			get {
				yield break;
			}
		}

		public virtual int StackConsumeCount {
			get {
				switch (Opcode.StackBehaviourPop) {
				case StackBehaviour.Pop0:
					return 0;

				case StackBehaviour.Pop1:
				case StackBehaviour.Popi:
				case StackBehaviour.Popref:
					return 1;

				case StackBehaviour.Pop1_pop1:
				case StackBehaviour.Popi_pop1:
				case StackBehaviour.Popi_popi:
				case StackBehaviour.Popi_popi8:
				case StackBehaviour.Popi_popr4:
				case StackBehaviour.Popi_popr8:
				case StackBehaviour.Popref_pop1:
				case StackBehaviour.Popref_popi:
					return 2;

				case StackBehaviour.Popi_popi_popi:
				case StackBehaviour.Popref_popi_pop1:
				case StackBehaviour.Popref_popi_popi:
				case StackBehaviour.Popref_popi_popi8:
				case StackBehaviour.Popref_popi_popr4:
				case StackBehaviour.Popref_popi_popr8:
				case StackBehaviour.Popref_popi_popref:
					return 3;

				default:
					throw new InvalidOperationException(string.Format("A StackBehaviourPop of \"{0}\" is unexpected.", Opcode.StackBehaviourPop));
				}
			}
		}

		public virtual int StackProduceCount {
			get {
				switch (Opcode.StackBehaviourPush) {
				case StackBehaviour.Push0:
					return 0;

				case StackBehaviour.Push1:
				case StackBehaviour.Pushi:
				case StackBehaviour.Pushi8:
				case StackBehaviour.Pushr4:
				case StackBehaviour.Pushr8:
				case StackBehaviour.Pushref:
					return 1;

				case StackBehaviour.Push1_push1:
					return 2;

				default:
					throw new InvalidOperationException(string.Format("A StackBehaviourPush of \"{0}\" is unexpected.", Opcode.StackBehaviourPush));
				}
			}
		}

		protected static string IndentString(int indent) {
			switch (indent) {
			case 0:
				return string.Empty;
			case 1:
				return "\t";
			case 2:
				return "\t\t";
			case 3:
				return "\t\t\t";
			case 4:
				return "\t\t\t\t";
			default:
				return new string('\t', indent);
			}
		}

		protected string StackName(int StackPointer) {
			System.Diagnostics.Debug.Assert(StackPointer > 0);
			return string.Format("stack_{0}", StackPointer - 1);
		}

		public static string LabelName(int Offset) {
			System.Diagnostics.Debug.Assert(Offset >= 0);
			return string.Format("IL_{0:X8}", Offset);
		}

		protected void WriteInstHeader(System.IO.TextWriter writer, int indent) {
			writer.WriteLine();
			writer.WriteLine("{0}// (stcon={3,2},stprd={4,2})\t{2}", IndentString(indent), Offset, ToString(), StackConsumeCount, StackProduceCount);
		}

		public abstract void WriteCode(System.IO.TextWriter writer, int indent, int CurStack);

		public abstract List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context);

		protected static sbyte ReadInt8(byte[] IL, int Offset) {
			return (sbyte)IL[Offset];
		}

		protected static byte ReadUInt8(byte[] IL, int Offset) {
			return IL[Offset];
		}

		protected static ushort ReadUInt16(byte[] IL, int Offset) {
			return (ushort)((ushort)IL[Offset + 0] | ((ushort)IL[Offset + 1] << 8));
		}

		protected static int ReadInt32(byte[] IL, int Offset) {
			return (int)((uint)IL[Offset + 0] | ((uint)IL[Offset + 1] << 8) | ((uint)IL[Offset + 2] << 16) | ((uint)IL[Offset + 3] << 24));
		}

		protected static float ReadFloat(byte[] IL, int Offset) {
			// TODO: endianness
			return BitConverter.ToSingle(IL, Offset);
		}

		protected static double ReadDouble(byte[] IL, int Offset) {
			// TODO: endianness
			return BitConverter.ToDouble(IL, Offset);
		}

		public delegate CilInstruction ConstructCilInstruction(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase);

		public static readonly Dictionary<OpCode, ConstructCilInstruction> Factory = GetCilFactoryMap();

		private static Dictionary<OpCode, ConstructCilInstruction> GetCilFactoryMap() {
			Dictionary<OpCode, ConstructCilInstruction> Map = new Dictionary<OpCode, ConstructCilInstruction>();
			Map.Add(OpCodes.Add, CilBinaryNumericInstruction.Create);

			Map.Add(OpCodes.Beq, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Beq_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bge, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bge_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bge_Un, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bge_Un_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bgt, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bgt_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bgt_Un, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bgt_Un_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Ble, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Ble_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Ble_Un, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Ble_Un_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Blt, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Blt_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Blt_Un, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Blt_Un_S, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bne_Un, CilComparisonAndBranchInstruction.Create);
			Map.Add(OpCodes.Bne_Un_S, CilComparisonAndBranchInstruction.Create);

			Map.Add(OpCodes.Br, CilBranchInstruction.Create);
			Map.Add(OpCodes.Br_S, CilBranchInstruction.Create);

			Map.Add(OpCodes.Brfalse, CilConditionalBranchInstruction.Create);
			Map.Add(OpCodes.Brfalse_S, CilConditionalBranchInstruction.Create);
			Map.Add(OpCodes.Brtrue, CilConditionalBranchInstruction.Create);
			Map.Add(OpCodes.Brtrue_S, CilConditionalBranchInstruction.Create);

			Map.Add(OpCodes.Call, CilCallInstruction.Create);

			Map.Add(OpCodes.Ceq, CilBinaryComparisonInstruction.Create);
			Map.Add(OpCodes.Cgt, CilBinaryComparisonInstruction.Create);
			Map.Add(OpCodes.Cgt_Un, CilBinaryComparisonInstruction.Create);
			Map.Add(OpCodes.Clt, CilBinaryComparisonInstruction.Create);
			Map.Add(OpCodes.Clt_Un, CilBinaryComparisonInstruction.Create);

			Map.Add(OpCodes.Conv_R4, CilConvertInstruction.Create_R4);
			Map.Add(OpCodes.Conv_R8, CilConvertInstruction.Create_R8);
			Map.Add(OpCodes.Conv_U, CilConvertInstruction.Create_U);

			Map.Add(OpCodes.Div, CilBinaryNumericInstruction.Create);
            Map.Add(OpCodes.Dup, CilDupInstruction.Create);

			Map.Add(OpCodes.Ldarg, CilLoadArgumentInstruction.Create);
			Map.Add(OpCodes.Ldarg_0, CilLoadArgumentInstruction.Create);
			Map.Add(OpCodes.Ldarg_1, CilLoadArgumentInstruction.Create);
			Map.Add(OpCodes.Ldarg_2, CilLoadArgumentInstruction.Create);
			Map.Add(OpCodes.Ldarg_3, CilLoadArgumentInstruction.Create);
			Map.Add(OpCodes.Ldarg_S, CilLoadArgumentInstruction.Create);

			Map.Add(OpCodes.Ldc_I4, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_0, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_1, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_2, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_3, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_4, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_5, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_6, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_7, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_8, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_M1, CilLoadI4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_I4_S, CilLoadI4ConstantInstruction.Create);

			Map.Add(OpCodes.Ldc_R4, CilLoadR4ConstantInstruction.Create);
			Map.Add(OpCodes.Ldc_R8, CilLoadR8ConstantInstruction.Create);

			Map.Add(OpCodes.Ldelem, CilLoadElementInstruction.CreateWithType);
			Map.Add(OpCodes.Ldelema, CilLoadElementAddressInstruction.CreateWithType);

			Map.Add(OpCodes.Ldelem_I, CilLoadElementInstruction.Create_I);
			Map.Add(OpCodes.Ldelem_I1, CilLoadElementInstruction.Create_I1);
			Map.Add(OpCodes.Ldelem_I2, CilLoadElementInstruction.Create_I2);
			Map.Add(OpCodes.Ldelem_I4, CilLoadElementInstruction.Create_I4);
			Map.Add(OpCodes.Ldelem_I8, CilLoadElementInstruction.Create_I8);
			Map.Add(OpCodes.Ldelem_R4, CilLoadElementInstruction.Create_R4);
			Map.Add(OpCodes.Ldelem_R8, CilLoadElementInstruction.Create_R8);
			Map.Add(OpCodes.Ldelem_Ref, CilLoadElementInstruction.Create_Ref);
			Map.Add(OpCodes.Ldelem_U1, CilLoadElementInstruction.Create_U1);
			Map.Add(OpCodes.Ldelem_U2, CilLoadElementInstruction.Create_U2);
			Map.Add(OpCodes.Ldelem_U4, CilLoadElementInstruction.Create_U4);

			Map.Add(OpCodes.Ldfld, CilLoadFieldInstruction.Create);
			Map.Add(OpCodes.Ldflda, CilLoadFieldAddressInstruction.Create);

			Map.Add(OpCodes.Ldloc, CilLoadLocalInstruction.Create);
			Map.Add(OpCodes.Ldloc_0, CilLoadLocalInstruction.Create);
			Map.Add(OpCodes.Ldloc_1, CilLoadLocalInstruction.Create);
			Map.Add(OpCodes.Ldloc_2, CilLoadLocalInstruction.Create);
			Map.Add(OpCodes.Ldloc_3, CilLoadLocalInstruction.Create);
			Map.Add(OpCodes.Ldloc_S, CilLoadLocalInstruction.Create);

            Map.Add(OpCodes.Ldobj, CilLoadObjectInstruction.Create);

			Map.Add(OpCodes.Ldsfld, CilLoadFieldInstruction.Create);
			Map.Add(OpCodes.Ldsflda, CilLoadFieldAddressInstruction.Create);

			Map.Add(OpCodes.Mul, CilBinaryNumericInstruction.Create);

			Map.Add(OpCodes.Neg, CilUnaryNumericInstruction.Create);

			Map.Add(OpCodes.Nop, CilNopInstruction.Create);

			Map.Add(OpCodes.Pop, CilPopInstruction.Create);

			Map.Add(OpCodes.Rem, CilBinaryNumericInstruction.Create);

			Map.Add(OpCodes.Ret, CilReturnInstruction.Create);

			Map.Add(OpCodes.Stelem_I, CilStoreElementInstruction.Create_I);
			Map.Add(OpCodes.Stelem_I1, CilStoreElementInstruction.Create_I1);
			Map.Add(OpCodes.Stelem_I2, CilStoreElementInstruction.Create_I2);
			Map.Add(OpCodes.Stelem_I4, CilStoreElementInstruction.Create_I4);
			Map.Add(OpCodes.Stelem_I8, CilStoreElementInstruction.Create_I8);
			Map.Add(OpCodes.Stelem_R4, CilStoreElementInstruction.Create_R4);
			Map.Add(OpCodes.Stelem_R8, CilStoreElementInstruction.Create_R8);
			Map.Add(OpCodes.Stelem_Ref, CilStoreElementInstruction.Create_Ref);

			Map.Add(OpCodes.Stloc, CilStoreLocalInstruction.Create);
			Map.Add(OpCodes.Stloc_0, CilStoreLocalInstruction.Create);
			Map.Add(OpCodes.Stloc_1, CilStoreLocalInstruction.Create);
			Map.Add(OpCodes.Stloc_2, CilStoreLocalInstruction.Create);
			Map.Add(OpCodes.Stloc_3, CilStoreLocalInstruction.Create);
			Map.Add(OpCodes.Stloc_S, CilStoreLocalInstruction.Create);

            Map.Add(OpCodes.Stobj, CilStoreObjectInstruction.Create);

			Map.Add(OpCodes.Sub, CilBinaryNumericInstruction.Create);

			return Map;
		}
	}

    public class CilDupInstruction : CilInstruction {
        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return new CilDupInstruction(Opcode, Offset);
		}

		private CilDupInstruction(OpCode Opcode, int Offset)
			: base(Opcode, Offset) {
			if (!(Opcode == OpCodes.Dup)) {
				throw new ArgumentException("Opcode");
			}
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = {2}", IndentString(indent), StackName(CurStack + 1), StackName(CurStack));
		}

		public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

            HighLevel.LocationNode Argument = Context.ReadStackLocationNode(Context.StackPointer);
            Context.DefineStackLocationNode(Context.StackPointer, Argument.Location.DataType);

			List.Add(new Ever.HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1), Argument));
			return List;
		}
	}

	public class CilUnaryNumericInstruction : CilInstruction {
		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return new CilUnaryNumericInstruction(Opcode, Offset);
		}

		private CilUnaryNumericInstruction(OpCode Opcode, int Offset)
			: base(Opcode, Offset) {
			if (!(Opcode == OpCodes.Neg)) {
				throw new ArgumentException("Opcode");
			}
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = negate({2})", IndentString(indent), StackName(CurStack), StackName(CurStack));
		}

		public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();
			List.Add(new Ever.HighLevel.AssignmentInstruction(Context.ReadStackLocationNode(Context.StackPointer), new HighLevel.NegNode(Context.ReadStackLocationNode(Context.StackPointer))));
			return List;
		}
	}

	public class CilBinaryNumericInstruction : CilInstruction {
		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return new CilBinaryNumericInstruction(Opcode, Offset);
		}

		private CilBinaryNumericInstruction(OpCode Opcode, int Offset)
			: base(Opcode, Offset) {
			if (!(Opcode == OpCodes.Add || Opcode == OpCodes.Sub || Opcode == OpCodes.Mul || Opcode == OpCodes.Div || Opcode == OpCodes.Rem)) {
				throw new ArgumentException("Opcode");
			}
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = {2} {3} {4}", IndentString(indent), StackName(CurStack - 1), StackName(CurStack - 1), GetOperatorSymbol(Opcode), StackName(CurStack));
		}

		public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

			HighLevel.Node Argument;
			HighLevel.Node Left = Context.ReadStackLocationNode(Context.StackPointer - 1), Right = Context.ReadStackLocationNode(Context.StackPointer);
			if (Opcode == OpCodes.Add) {
				Argument = new HighLevel.AddNode(Left, Right);
			} else if (Opcode == OpCodes.Sub) {
				Argument = new HighLevel.SubNode(Left, Right);
			} else if (Opcode == OpCodes.Mul) {
				Argument = new HighLevel.MulNode(Left, Right);
			} else if (Opcode == OpCodes.Div) {
				Argument = new HighLevel.DivNode(Left, Right);
			} else if (Opcode == OpCodes.Rem) {
				Argument = new HighLevel.ModNode(Left, Right);
			} else {
				throw new InvalidOperationException();
			}

			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer - 1), Argument));
			return List;
		}

		public static string GetOperatorSymbol(OpCode Opcode) {
			if (Opcode == OpCodes.Add) {
				return "+";
			} else if (Opcode == OpCodes.Sub) {
				return "-";
			} else if (Opcode == OpCodes.Mul) {
				return "*";
			} else if (Opcode == OpCodes.Div) {
				return "/";
			} else if (Opcode == OpCodes.Rem) {
				return "%";
			} else {
				throw new InvalidOperationException();
			}
		}
	}

	public class CilBinaryComparisonInstruction : CilInstruction {
		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return new CilBinaryComparisonInstruction(Opcode, Offset);
		}

		private CilBinaryComparisonInstruction(OpCode Opcode, int Offset)
			: base(Opcode, Offset) {
			if (!(Opcode == OpCodes.Ceq || Opcode == OpCodes.Cgt || Opcode == OpCodes.Cgt_Un || Opcode == OpCodes.Clt || Opcode == OpCodes.Clt_Un)) {
				throw new ArgumentException("Opcode");
			}
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = ({2} {3} {4}) ? 1 : 0", IndentString(indent), StackName(CurStack - 1), StackName(CurStack - 1), GetOperatorSymbol(Opcode), StackName(CurStack));
		}

		public static string GetOperatorSymbol(OpCode Opcode) {
			if (Opcode == OpCodes.Ceq) {
				return "==";
			} else if (Opcode == OpCodes.Cgt) {
				return "s>";
			} else if (Opcode == OpCodes.Cgt_Un) {
				return "u>";
			} else if (Opcode == OpCodes.Clt) {
				return "s<";
			} else if (Opcode == OpCodes.Clt_Un) {
				return "u<";
			} else {
				throw new InvalidOperationException();
			}
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			HighLevel.Node Argument;
			if (Opcode == OpCodes.Ceq) {
				Argument = new HighLevel.EqualsNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else if (Opcode == OpCodes.Cgt || Opcode == OpCodes.Cgt_Un) {
				Argument = new HighLevel.GreaterNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else if (Opcode == OpCodes.Clt || Opcode == OpCodes.Clt_Un) {
				Argument = new HighLevel.LessNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else {
				throw new InvalidOperationException();
			}

			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer - 1, typeof(int)), Argument));
			return List;
		}
	}

	public class CilComparisonAndBranchInstruction : CilInstruction {
		private int m_BranchTargetOffset;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int BranchTarget;
			if (Opcode.OperandType == OperandType.ShortInlineBrTarget) {
				BranchTarget = NextOffset + ReadInt8(IL, Offset + Opcode.Size);
			} else if (Opcode.OperandType == OperandType.InlineBrTarget) {
				BranchTarget = NextOffset + ReadInt32(IL, Offset + Opcode.Size);
			} else {
				throw new ArgumentException("Opcode \"" + Opcode.ToString() + "\" invalid for CilBinaryComparisonAndBranchInstruction.");
			}

			return new CilComparisonAndBranchInstruction(Opcode, Offset, BranchTarget);
		}

		private CilComparisonAndBranchInstruction(OpCode Opcode, int Offset, int BranchTargetOffset)
			: base(Opcode, Offset) {
			if (!(Opcode == OpCodes.Beq || Opcode == OpCodes.Beq_S ||
				Opcode == OpCodes.Bge || Opcode == OpCodes.Bge_S ||	Opcode == OpCodes.Bge_Un || Opcode == OpCodes.Bge_Un_S ||
				Opcode == OpCodes.Bgt || Opcode == OpCodes.Bgt_S || Opcode == OpCodes.Bgt_Un || Opcode == OpCodes.Bgt_Un_S ||
				Opcode == OpCodes.Ble || Opcode == OpCodes.Ble_S || Opcode == OpCodes.Ble_Un || Opcode == OpCodes.Ble_Un_S ||
				Opcode == OpCodes.Blt || Opcode == OpCodes.Blt_S || Opcode == OpCodes.Blt_Un || Opcode == OpCodes.Blt_Un_S ||
				Opcode == OpCodes.Bne_Un || Opcode == OpCodes.Bne_Un_S
				)) {
				throw new ArgumentException("Opcode");
			}

			m_BranchTargetOffset = BranchTargetOffset;
		}

		public override IEnumerable<int> BranchTargetOffsets {
			get {
				yield return m_BranchTargetOffset;
			}
		}

		public override string ToString() {
			return base.ToString() + " IL_" + m_BranchTargetOffset.ToString("X4");
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}if ({1} {2} {3}) goto {4}", IndentString(indent), StackName(CurStack - 1), GetOperatorSymbol(Opcode), StackName(CurStack), LabelName(m_BranchTargetOffset));
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			HighLevel.Node Argument;
			if (Opcode == OpCodes.Beq || Opcode == OpCodes.Beq_S) {
				Argument = new HighLevel.EqualsNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else if (Opcode == OpCodes.Bne_Un || Opcode == OpCodes.Bne_Un_S) {
				Argument = new HighLevel.EqualsNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else if (Opcode == OpCodes.Bge || Opcode == OpCodes.Bge_S || Opcode == OpCodes.Bge_Un || Opcode == OpCodes.Bge_Un_S) {
				Argument = new HighLevel.GreaterEqualsNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else if (Opcode == OpCodes.Bgt || Opcode == OpCodes.Bgt_S || Opcode == OpCodes.Bgt_Un || Opcode == OpCodes.Bgt_Un_S) {
				Argument = new HighLevel.GreaterNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else if (Opcode == OpCodes.Ble || Opcode == OpCodes.Ble_S || Opcode == OpCodes.Ble_Un || Opcode == OpCodes.Ble_Un_S) {
				Argument = new HighLevel.LessEqualsNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else if (Opcode == OpCodes.Blt || Opcode == OpCodes.Blt_S || Opcode == OpCodes.Blt_Un || Opcode == OpCodes.Blt_Un_S) {
				Argument = new HighLevel.LessNode(Context.ReadStackLocationNode(Context.StackPointer - 1), Context.ReadStackLocationNode(Context.StackPointer));
			} else {
				throw new InvalidOperationException();
			}

			List.Add(new HighLevel.ConditionalBranchInstruction(Argument, Context.GetBlock(m_BranchTargetOffset)));
			return List;
		}

		public static string GetOperatorSymbol(OpCode Opcode) {
			if (Opcode == OpCodes.Beq || Opcode == OpCodes.Beq_S) {
				return "==";
			} else if (Opcode == OpCodes.Bne_Un || Opcode == OpCodes.Bne_Un_S) {
				return "u!=";
			} else if (Opcode == OpCodes.Bge || Opcode == OpCodes.Bge_S) {
				return "s>=";
			} else if (Opcode == OpCodes.Bge_Un || Opcode == OpCodes.Bge_Un_S) {
				return "u>=";
			} else if (Opcode == OpCodes.Bgt || Opcode == OpCodes.Bgt_S) {
				return "s>";
			} else if (Opcode == OpCodes.Bgt_Un || Opcode == OpCodes.Bgt_Un_S) {
				return "u>";
			} else if (Opcode == OpCodes.Ble || Opcode == OpCodes.Ble_S) {
				return "s<=";
			} else if (Opcode == OpCodes.Ble_Un || Opcode == OpCodes.Ble_Un_S) {
				return "u<=";
			} else if (Opcode == OpCodes.Blt || Opcode == OpCodes.Blt_S) {
				return "s<";
			} else if (Opcode == OpCodes.Blt_Un || Opcode == OpCodes.Blt_Un_S) {
				return "u<";
			} else {
				throw new InvalidOperationException();
			}
		}
	}

	public class CilBranchInstruction : CilInstruction {
		private int m_BranchTargetOffset;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int BranchTarget;
			if (Opcode.OperandType == OperandType.ShortInlineBrTarget) {
				BranchTarget = NextOffset + ReadInt8(IL, Offset + Opcode.Size);
			} else if (Opcode.OperandType == OperandType.InlineBrTarget) {
				BranchTarget = NextOffset + ReadInt32(IL, Offset + Opcode.Size);
			} else {
				throw new ArgumentException("Opcode \"" + Opcode.ToString() + "\" invalid for CilConditionalBranchInstruction.");
			}

			return new CilBranchInstruction(Opcode, Offset, BranchTarget);
		}

		private CilBranchInstruction(OpCode Opcode, int Offset, int BranchTargetOffset)
			: base(Opcode, Offset) {
			m_BranchTargetOffset = BranchTargetOffset;
		}

		public override bool CanFallThrough {
			get {
				return false;
			}
		}

		public override IEnumerable<int> BranchTargetOffsets {
			get {
				yield return m_BranchTargetOffset;
			}
		}

		public override string ToString() {
			return base.ToString() + " IL_" + m_BranchTargetOffset.ToString("X4");
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}goto {1};", IndentString(indent), LabelName(m_BranchTargetOffset));
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();
			List.Add(new HighLevel.BranchInstruction(Context.GetBlock(m_BranchTargetOffset)));
			return List;
		}
	}

	public class CilConditionalBranchInstruction : CilInstruction {
		private int m_BranchTargetOffset;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int BranchTarget;
			if (Opcode.OperandType == OperandType.ShortInlineBrTarget) {
				BranchTarget = NextOffset + ReadInt8(IL, Offset + Opcode.Size);
			} else if (Opcode.OperandType == OperandType.InlineBrTarget) {
				BranchTarget = NextOffset + ReadInt32(IL, Offset + Opcode.Size);
			} else {
				throw new ArgumentException("Opcode \"" + Opcode.ToString() + "\" invalid for CilConditionalBranchInstruction.");
			}

			return new CilConditionalBranchInstruction(Opcode, Offset, BranchTarget);
		}

		private CilConditionalBranchInstruction(OpCode Opcode, int Offset, int BranchTargetOffset)
			: base(Opcode, Offset) {
			if (!(Opcode == OpCodes.Brtrue || Opcode == OpCodes.Brtrue_S || Opcode == OpCodes.Brfalse || Opcode == OpCodes.Brfalse_S)) {
				throw new ArgumentException("Opcode");
			}

			m_BranchTargetOffset = BranchTargetOffset;
		}

		public override IEnumerable<int> BranchTargetOffsets {
			get {
				yield return m_BranchTargetOffset;
			}
		}

		public override string ToString() {
			return base.ToString() + " IL_" + m_BranchTargetOffset.ToString("X4");
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}if ({1}{2}) goto {3};", IndentString(indent), (Opcode == OpCodes.Brfalse || Opcode == OpCodes.Brfalse_S) ? "!" : string.Empty, StackName(CurStack), LabelName(m_BranchTargetOffset));
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();

			HighLevel.Node Argument = Context.ReadStackLocationNode(Context.StackPointer);
			if (Opcode == OpCodes.Brfalse || Opcode == OpCodes.Brfalse_S) {
				Argument = new HighLevel.LogicalNotNode(Argument);
			}

			List.Add(new HighLevel.ConditionalBranchInstruction(Argument, Context.GetBlock(m_BranchTargetOffset)));
			return List;
		}
	}

	public class CilCallInstruction : CilInstruction {
		private MethodInfo m_MethodInfo;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int Token = ReadInt32(IL, Offset + Opcode.Size);
			MethodBase TargetMethodBase = ParentMethodBase.Module.ResolveMethod(Token);

			return new CilCallInstruction(Opcode, Offset, TargetMethodBase);
		}

		private CilCallInstruction(OpCode Opcode, int Offset, MethodBase MethodBase)
			: base(Opcode, Offset) {
			if (Opcode != OpCodes.Call) {
				throw new ArgumentException("Unsupported Opcode " + Opcode.ToString());
			} else if (MethodBase == null || MethodBase.IsConstructor) {
				throw new ArgumentNullException("MethodBase");
			}

			System.Diagnostics.Debug.Assert(MethodBase is MethodInfo);
			m_MethodInfo = (MethodInfo)MethodBase;
		}

		public MethodInfo MethodInfo {
			get {
				return m_MethodInfo;
			}
		}

		public override string ToString() {
			return base.ToString() + " " + m_MethodInfo.ToString();
		}

		public override int StackConsumeCount {
			get {
				System.Diagnostics.Debug.Assert(Opcode.StackBehaviourPop == StackBehaviour.Varpop);

				int ConsumeCount = 0;
				if ((MethodInfo.CallingConvention & CallingConventions.HasThis) != 0) {
					ConsumeCount++;
				}

				ConsumeCount += MethodInfo.GetParameters().Length;
				return ConsumeCount;
			}
		}

		public override int StackProduceCount {
			get {
				System.Diagnostics.Debug.Assert(Opcode.StackBehaviourPush == StackBehaviour.Varpush);

				int ProduceCount = 0;
				if (MethodInfo.ReturnType != typeof(void)) {
					ProduceCount++;
				}

				return ProduceCount;
			}
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.Write("{0}", IndentString(indent));
			if (StackProduceCount == 1) {
				writer.Write("{0} = ", StackName(CurStack - StackConsumeCount + 1));
			} else {
				System.Diagnostics.Debug.Assert(StackProduceCount == 0);
			}

			writer.Write("cilfn[\"{0}\"](", m_MethodInfo.DeclaringType.ToString() + "::" + m_MethodInfo.ToString());
			for (int i = StackConsumeCount - 1; i >= 0; i--) {
				writer.Write(StackName(CurStack - i));
			}
			writer.WriteLine(");");
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			HighLevel.Node Result = null;
			if (StackProduceCount != 0) {
				Result = Context.DefineStackLocationNode(Context.StackPointer - StackConsumeCount + 1, MethodInfo.ReturnType);
			}

			HighLevel.Node CallNode = new HighLevel.CallNode(MethodInfo);
			for (int i = StackConsumeCount - 1; i >= 0; i--) {
				CallNode.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer - i));
			}

			Context.TranslateCallNode(ref Result, ref CallNode);

			List.Add(new HighLevel.AssignmentInstruction(Result, CallNode));
			return List;
		}
	}

	public class CilNopInstruction : CilInstruction {
		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return new CilNopInstruction(Opcode, Offset);
		}

		private CilNopInstruction(OpCode Opcode, int Offset)
			: base(Opcode, Offset) {
			if (Opcode != OpCodes.Nop) {
				throw new ArgumentException("Opcode");
			}
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}// CIL NOP", IndentString(indent));
		}

		public override List<HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			return new List<HighLevel.Instruction>();
		}
	}

	public class CilPopInstruction : CilInstruction {
		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return new CilPopInstruction(Opcode, Offset);
		}

		private CilPopInstruction(OpCode Opcode, int Offset)
			: base(Opcode, Offset) {
			if (Opcode != OpCodes.Pop) {
				throw new ArgumentException("Opcode");
			}
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}// CIL POP", IndentString(indent));
		}

		public override List<HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			return new List<HighLevel.Instruction>();
		}
	}

	public class CilLoadI4ConstantInstruction : CilInstruction {
		private int m_Constant;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int? Constant = null;
			if (Opcode == OpCodes.Ldc_I4) {
				Constant = ReadInt32(IL, Offset + Opcode.Size);
			} else if (Opcode == OpCodes.Ldc_I4_S) {
				Constant = ReadInt8(IL, Offset + Opcode.Size);
			}

			return new CilLoadI4ConstantInstruction(Opcode, Offset, Constant);
		}

		private CilLoadI4ConstantInstruction(OpCode Opcode, int Offset, int? Constant)
			: base(Opcode, Offset) {
			if (Opcode == OpCodes.Ldc_I4_0) {
				Constant = 0;
			} else if (Opcode == OpCodes.Ldc_I4_1) {
				Constant = 1;
			} else if (Opcode == OpCodes.Ldc_I4_2) {
				Constant = 2;
			} else if (Opcode == OpCodes.Ldc_I4_3) {
				Constant = 3;
			} else if (Opcode == OpCodes.Ldc_I4_4) {
				Constant = 4;
			} else if (Opcode == OpCodes.Ldc_I4_5) {
				Constant = 5;
			} else if (Opcode == OpCodes.Ldc_I4_6) {
				Constant = 6;
			} else if (Opcode == OpCodes.Ldc_I4_7) {
				Constant = 7;
			} else if (Opcode == OpCodes.Ldc_I4_8) {
				Constant = 8;
			} else if (Opcode == OpCodes.Ldc_I4_M1) {
				Constant = -1;
			}
			System.Diagnostics.Debug.Assert(Constant.HasValue);
			m_Constant = Constant.Value;
		}

		public override string ToString() {
			return base.ToString() + (Opcode == OpCodes.Ldc_I4 || Opcode == OpCodes.Ldc_I4_S ? " " + m_Constant.ToString() : string.Empty);
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), m_Constant);
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();
			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1, typeof(int)), new HighLevel.IntegerConstantNode(m_Constant)));
			return List;
		}
	}

	public class CilLoadR4ConstantInstruction : CilInstruction {
		private float m_Constant;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return new CilLoadR4ConstantInstruction(Opcode, Offset, ReadFloat(IL, Offset + Opcode.Size));
		}

		private CilLoadR4ConstantInstruction(OpCode Opcode, int Offset, float Constant)
			: base(Opcode, Offset) {
			if (Opcode != OpCodes.Ldc_R4) {
				throw new ArgumentException();
			}
			m_Constant = Constant;
		}

		public float Constant {
			get {
				return m_Constant;
			}
		}

		public override string ToString() {
			return base.ToString() + " " + m_Constant.ToString("F1") + "f";
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), m_Constant);
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();
			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1, typeof(float)), new HighLevel.FloatConstantNode(m_Constant)));
			return List;
		}
	}

	public class CilLoadR8ConstantInstruction : CilInstruction {
		private double m_Constant;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return new CilLoadR8ConstantInstruction(Opcode, Offset, ReadDouble(IL, Offset + Opcode.Size));
		}

		private CilLoadR8ConstantInstruction(OpCode Opcode, int Offset, double Constant)
			: base(Opcode, Offset) {
			if (Opcode != OpCodes.Ldc_R8) {
				throw new ArgumentException();
			}
			m_Constant = Constant;
		}

		public double Constant {
			get {
				return m_Constant;
			}
		}

		public override string ToString() {
			return base.ToString() + " " + m_Constant.ToString("F1");
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), m_Constant);
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();
			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1, typeof(double)), new HighLevel.DoubleConstantNode(m_Constant)));
			return List;
		}
	}

	public class CilLoadLocalInstruction : CilInstruction {
		private int m_Index;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int Index;
			if (Opcode == OpCodes.Ldloc_S) {
				Index = ReadUInt8(IL, Offset + Opcode.Size);
			} else if (Opcode == OpCodes.Ldloc) {
				Index = ReadUInt16(IL, Offset + Opcode.Size);
			} else {
				Index = -1;
			}

			return new CilLoadLocalInstruction(Opcode, Offset, Index);
		}

		private CilLoadLocalInstruction(OpCode Opcode, int Offset, int Index)
			: base(Opcode, Offset) {
			if (Opcode == OpCodes.Ldloc_0) {
				m_Index = 0;
			} else if (Opcode == OpCodes.Ldloc_1) {
				m_Index = 1;
			} else if (Opcode == OpCodes.Ldloc_2) {
				m_Index = 2;
			} else if (Opcode == OpCodes.Ldloc_3) {
				m_Index = 3;
			} else if (Opcode == OpCodes.Ldloc || Opcode == OpCodes.Ldloc_S) {
				m_Index = Index;
			} else {
				throw new ArgumentException("Opcode");
			}

			System.Diagnostics.Debug.Assert(m_Index >= 0);
		}

		public override string ToString() {
			return base.ToString() + ((Opcode == OpCodes.Ldloc || Opcode == OpCodes.Ldloc_S) ? " " + m_Index : string.Empty);
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), "local_" + m_Index.ToString());
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1), Context.LocalVariableNode(m_Index)));
			return List;
		}
	}

	public class CilStoreLocalInstruction : CilInstruction {
		private int m_Index;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int Index;
			if (Opcode == OpCodes.Stloc_S) {
				Index = ReadUInt8(IL, Offset + Opcode.Size);
			} else if (Opcode == OpCodes.Stloc) {
				Index = ReadUInt16(IL, Offset + Opcode.Size);
			} else {
				Index = -1;
			}

			return new CilStoreLocalInstruction(Opcode, Offset, Index);
		}

		private CilStoreLocalInstruction(OpCode Opcode, int Offset, int Index)
			: base(Opcode, Offset) {
			if (Opcode == OpCodes.Stloc_0) {
				m_Index = 0;
			} else if (Opcode == OpCodes.Stloc_1) {
				m_Index = 1;
			} else if (Opcode == OpCodes.Stloc_2) {
				m_Index = 2;
			} else if (Opcode == OpCodes.Stloc_3) {
				m_Index = 3;
			} else if (Opcode == OpCodes.Stloc || Opcode == OpCodes.Stloc_S) {
				m_Index = Index;
			} else {
				throw new ArgumentException();
			}

			System.Diagnostics.Debug.Assert(m_Index >= 0);
		}

		public override string ToString() {
			return base.ToString() + ((Opcode == OpCodes.Stloc || Opcode == OpCodes.Stloc_S) ? " " + m_Index : string.Empty);
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = {2};", IndentString(indent), "local_" + m_Index.ToString(), StackName(CurStack));
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();
			List.Add(new HighLevel.AssignmentInstruction(Context.LocalVariableNode(m_Index), Context.ReadStackLocationNode(Context.StackPointer)));
			return List;
		}
	}

	public class CilLoadArgumentInstruction : CilInstruction {
		private int m_Index;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int Index;
			if (Opcode == OpCodes.Ldarg_S) {
				Index = IL[Offset + Opcode.Size + 0];
			} else if (Opcode == OpCodes.Ldarg) {
				Index = ReadUInt16(IL, Offset + Opcode.Size);
			} else {
				Index = -1;
			}

			return new CilLoadArgumentInstruction(Opcode, Offset, Index);
		}

		private CilLoadArgumentInstruction(OpCode Opcode, int Offset, int Index)
			: base(Opcode, Offset) {
			if (Opcode == OpCodes.Ldarg_0) {
				m_Index = 0;
			} else if (Opcode == OpCodes.Ldarg_1) {
				m_Index = 1;
			} else if (Opcode == OpCodes.Ldarg_2) {
				m_Index = 2;
			} else if (Opcode == OpCodes.Ldarg_3) {
				m_Index = 3;
			} else if (Opcode == OpCodes.Ldarg || Opcode == OpCodes.Ldarg_S) {
				m_Index = Index;
			} else {
				throw new ArgumentException();
			}

			System.Diagnostics.Debug.Assert(m_Index >= 0);
		}

		public override string ToString() {
			return base.ToString() + ((Opcode == OpCodes.Ldarg || Opcode == OpCodes.Ldarg_S) ? " " + m_Index : string.Empty);
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = {2};", IndentString(indent), StackName(CurStack + 1), "argument_" + m_Index.ToString());
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();
			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer + 1), Context.ArgumentNode(m_Index)));
			return List;
		}
	}

	public class CilConvertInstruction : CilInstruction {
		private Type m_Type;

		public static CilInstruction Create_I(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(IntPtr));
		}

		public static CilInstruction Create_I1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(sbyte));
		}

		public static CilInstruction Create_I2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(short));
		}

		public static CilInstruction Create_I4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(int));
		}

		public static CilInstruction Create_I8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(long));
		}

		public static CilInstruction Create_U(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(UIntPtr));
		}

		public static CilInstruction Create_U1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(byte));
		}

		public static CilInstruction Create_U2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(ushort));
		}

		public static CilInstruction Create_U4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(uint));
		}

		public static CilInstruction Create_R4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(float));
		}

		public static CilInstruction Create_R8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(double));
		}

		private static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase, Type Type) {
			return new CilConvertInstruction(Opcode, Offset, Type);
		}

		private CilConvertInstruction(OpCode Opcode, int Offset, Type Type)
			: base(Opcode, Offset) {
			m_Type = Type;
		}

		public Type Type {
			get {
				return m_Type;
			}
		}

		public override string ToString() {
			return base.ToString() + " " + m_Type.ToString();
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = ({2})[{3}];", IndentString(indent), StackName(CurStack), m_Type.ToString(), StackName(CurStack));
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer), new HighLevel.CastNode(Context.ReadStackLocationNode(Context.StackPointer), Type)));
			return List;
		}
	}

	public class CilLoadElementInstruction : CilInstruction {
		private Type m_ArrayType;

		public static CilInstruction CreateWithType(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int Token = ReadInt32(IL, Offset + Opcode.Size);
			Type ElementType = ParentMethodBase.Module.ResolveType(Token);
			Type ArrayType = Array.CreateInstance(ElementType, 1).GetType();
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, ArrayType);
		}

		public static CilInstruction Create_I(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(IntPtr[]));
		}

		public static CilInstruction Create_I1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(sbyte[]));
		}

		public static CilInstruction Create_I2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(short[]));
		}

		public static CilInstruction Create_I4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(int[]));
		}

		public static CilInstruction Create_I8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(long[]));
		}

		public static CilInstruction Create_U1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(byte[]));
		}

		public static CilInstruction Create_U2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(ushort[]));
		}

		public static CilInstruction Create_U4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(uint[]));
		}

		public static CilInstruction Create_R4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(float[]));
		}

		public static CilInstruction Create_R8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(double[]));
		}

		public static CilInstruction Create_Ref(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(object[]));
		}

		private static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase, Type ArrayType) {
			return new CilLoadElementInstruction(Opcode, Offset, ArrayType);
		}

		private CilLoadElementInstruction(OpCode Opcode, int Offset, Type ArrayType)
			: base(Opcode, Offset) {
			System.Diagnostics.Debug.Assert(ArrayType.IsArray);
			m_ArrayType = ArrayType;
		}

		public Type ArrayType {
			get {
				return m_ArrayType;
			}
		}

		public override string ToString() {
			return base.ToString() + " " + m_ArrayType.GetElementType();
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = ({2})[{3}];", IndentString(indent), StackName(CurStack - 1), StackName(CurStack - 1), StackName(CurStack));
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			HighLevel.ArrayAccessNode Argument = new Ever.HighLevel.ArrayAccessNode(ArrayType);
			Argument.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer - 1));
			Argument.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer));

			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer - 1), Argument));
			return List;
		}
	}

	public class CilLoadElementAddressInstruction : CilInstruction {
		private Type m_ArrayType;

		public static CilInstruction CreateWithType(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int Token = ReadInt32(IL, Offset + Opcode.Size);
			Type ElementType = ParentMethodBase.Module.ResolveType(Token);
			Type ArrayType = Array.CreateInstance(ElementType, 1).GetType();
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, ArrayType);
		}

		private static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase, Type ArrayType) {
			return new CilLoadElementAddressInstruction(Opcode, Offset, ArrayType);
		}

		private CilLoadElementAddressInstruction(OpCode Opcode, int Offset, Type ArrayType)
			: base(Opcode, Offset) {
			System.Diagnostics.Debug.Assert(ArrayType.IsArray);
			m_ArrayType = ArrayType;
		}

		public Type ArrayType {
			get {
				return m_ArrayType;
			}
		}

		public override string ToString() {
			return base.ToString() + " " + m_ArrayType.GetElementType();
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = &(({2})[{3}]);", IndentString(indent), StackName(CurStack - 1), StackName(CurStack - 1), StackName(CurStack));
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			HighLevel.ArrayAccessNode Argument = new Ever.HighLevel.ArrayAccessNode(ArrayType);
			Argument.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer - 1));
			Argument.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer));

			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer - 1), new HighLevel.AddressOfNode(Argument)));
			return List;
		}
	}

	public class CilStoreElementInstruction : CilInstruction {
		private Type m_ArrayType;

		public static CilInstruction CreateWithType(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int Token = ReadInt32(IL, Offset + Opcode.Size);
			Type ElementType = ParentMethodBase.Module.ResolveType(Token);
			Type ArrayType = Array.CreateInstance(ElementType, 1).GetType();
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, ArrayType);
		}

		public static CilInstruction Create_I(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(IntPtr[]));
		}

		public static CilInstruction Create_I1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(sbyte[]));
		}

		public static CilInstruction Create_I2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(short[]));
		}

		public static CilInstruction Create_I4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(int[]));
		}

		public static CilInstruction Create_I8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(long[]));
		}

		public static CilInstruction Create_U1(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(byte[]));
		}

		public static CilInstruction Create_U2(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(ushort[]));
		}

		public static CilInstruction Create_U4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(uint[]));
		}

		public static CilInstruction Create_R4(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(float[]));
		}

		public static CilInstruction Create_R8(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(double[]));
		}

		public static CilInstruction Create_Ref(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			return Create(Opcode, IL, Offset, NextOffset, ParentMethodBase, typeof(object[]));
		}

		private static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase, Type ArrayType) {
			return new CilStoreElementInstruction(Opcode, Offset, ArrayType);
		}

		private CilStoreElementInstruction(OpCode Opcode, int Offset, Type ArrayType)
			: base(Opcode, Offset) {
			System.Diagnostics.Debug.Assert(ArrayType.IsArray);
			m_ArrayType = ArrayType;
		}

		public Type ArrayType {
			get {
				return m_ArrayType;
			}
		}

		public override string ToString() {
			return base.ToString() + " " + m_ArrayType.GetElementType();
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}({1})[{2}] = {3};", IndentString(indent), StackName(CurStack - 2), StackName(CurStack - 1), StackName(CurStack));
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			HighLevel.ArrayAccessNode Argument = new Ever.HighLevel.ArrayAccessNode(ArrayType);
			Argument.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer - 2));
			Argument.SubNodes.Add(Context.ReadStackLocationNode(Context.StackPointer - 1));

			List.Add(new HighLevel.AssignmentInstruction(Argument, Context.ReadStackLocationNode(Context.StackPointer)));
			return List;
		}
	}

    public class CilLoadObjectInstruction : CilInstruction {
        private Type m_Type;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
            int Token = ReadInt32(IL, Offset + Opcode.Size);
            Type Type = ParentMethodBase.Module.ResolveType(Token);
            return new CilLoadObjectInstruction(Opcode, Offset, Type);
        }

        private CilLoadObjectInstruction(OpCode Opcode, int Offset, Type Type)
            : base(Opcode, Offset) {
            if (Opcode != OpCodes.Ldobj) {
                throw new ArgumentException("Opcode");
            }
            m_Type = Type;
        }

        public Type Type {
            get {
                return m_Type;
            }
        }

        public override string ToString() {
            return base.ToString() + " " + m_Type.ToString();
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = *({2});", IndentString(indent), StackName(CurStack), StackName(CurStack));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context) {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer), new HighLevel.DerefNode(Context.ReadStackLocationNode(Context.StackPointer))));
            return List;
        }
    }

    public class CilStoreObjectInstruction : CilInstruction {
        private Type m_Type;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
            int Token = ReadInt32(IL, Offset + Opcode.Size);
            Type Type = ParentMethodBase.Module.ResolveType(Token);
            return new CilStoreObjectInstruction(Opcode, Offset, Type);
        }

        private CilStoreObjectInstruction(OpCode Opcode, int Offset, Type Type)
            : base(Opcode, Offset) {
            if (Opcode != OpCodes.Stobj) {
                throw new ArgumentException("Opcode");
            }
            m_Type = Type;
        }

        public Type Type {
            get {
                return m_Type;
            }
        }

        public override string ToString() {
            return base.ToString() + " " + m_Type.ToString();
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}*({1}) = {2};", IndentString(indent), StackName(CurStack - 1), StackName(CurStack));
        }

        public override List<HighLevel.Instruction> GetHighLevel(HighLevel.HlGraph Context) {
            List<HighLevel.Instruction> List = new List<HighLevel.Instruction>();
            List.Add(new HighLevel.AssignmentInstruction(new HighLevel.DerefNode(Context.ReadStackLocationNode(Context.StackPointer - 1)), Context.ReadStackLocationNode(Context.StackPointer)));
            return List;
        }
    }

    public class CilLoadFieldInstruction : CilInstruction {
        private FieldInfo m_FieldInfo;

        public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
            int Token = ReadInt32(IL, Offset + Opcode.Size);
            FieldInfo FieldInfo = ParentMethodBase.Module.ResolveField(Token);
            return new CilLoadFieldInstruction(Opcode, Offset, FieldInfo);
        }

        private CilLoadFieldInstruction(OpCode Opcode, int Offset, FieldInfo FieldInfo)
            : base(Opcode, Offset) {
            if (Opcode != OpCodes.Ldfld) {
                throw new ArgumentException("Opcode");
            }
            m_FieldInfo = FieldInfo;
        }

        public FieldInfo FieldInfo {
            get {
                return m_FieldInfo;
            }
        }

        public override string ToString() {
            return base.ToString() + " " + m_FieldInfo.DeclaringType.FullName + "::" + m_FieldInfo.Name;
        }

        public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
            WriteInstHeader(writer, indent);
            writer.WriteLine("{0}{1} = ({2}).__field(\"{3}\");", IndentString(indent), StackName(CurStack), StackName(CurStack), m_FieldInfo.DeclaringType.FullName + "::" + m_FieldInfo.Name);
        }

        public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
            List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

            HighLevel.Node Argument;
            if (FieldInfo.IsStatic) {
                Argument = new HighLevel.LocationNode(Context.StaticFieldLocation(FieldInfo));
            } else {
                Argument = new HighLevel.InstanceFieldNode(Context.ReadStackLocationNode(Context.StackPointer), FieldInfo);
            }

            List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer), Argument));
            return List;
        }
    }

	public class CilLoadFieldAddressInstruction : CilInstruction {
		private FieldInfo m_FieldInfo;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			int Token = ReadInt32(IL, Offset + Opcode.Size);
			FieldInfo FieldInfo = ParentMethodBase.Module.ResolveField(Token);
			return new CilLoadFieldAddressInstruction(Opcode, Offset, FieldInfo);
		}

		private CilLoadFieldAddressInstruction(OpCode Opcode, int Offset, FieldInfo FieldInfo)
			: base(Opcode, Offset) {
			if (Opcode != OpCodes.Ldflda) {
				throw new ArgumentException("Opcode");
			}
			m_FieldInfo = FieldInfo;
		}

		public FieldInfo FieldInfo {
			get {
				return m_FieldInfo;
			}
		}

		public override string ToString() {
			return base.ToString() + " " + m_FieldInfo.DeclaringType.FullName + "::" + m_FieldInfo.Name;
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);
			writer.WriteLine("{0}{1} = &({2}).__field(\"{3}\");", IndentString(indent), StackName(CurStack), StackName(CurStack), m_FieldInfo.DeclaringType.FullName + "::" + m_FieldInfo.Name);
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();

			HighLevel.Node Argument;
			if (FieldInfo.IsStatic) {
				Argument = new HighLevel.LocationNode(Context.StaticFieldLocation(FieldInfo));
			} else {
				Argument = new HighLevel.InstanceFieldNode(Context.ReadStackLocationNode(Context.StackPointer), FieldInfo);
			}

			if (!FieldInfo.FieldType.IsValueType) {
				Argument = new HighLevel.AddressOfNode(Argument);
			}

			List.Add(new HighLevel.AssignmentInstruction(Context.DefineStackLocationNode(Context.StackPointer), Argument));
			return List;
		}
	}

	public class CilReturnInstruction : CilInstruction {
		private Type m_ReturnType;

		public static CilInstruction Create(OpCode Opcode, byte[] IL, int Offset, int NextOffset, MethodBase ParentMethodBase) {
			Type ReturnType = typeof(void);
			if (!ParentMethodBase.IsConstructor) {
				System.Diagnostics.Debug.Assert(ParentMethodBase is MethodInfo);
				ReturnType = ((MethodInfo)ParentMethodBase).ReturnType;
			}

			return new CilReturnInstruction(Opcode, Offset, ReturnType);
		}

		private CilReturnInstruction(OpCode Opcode, int Offset, Type ReturnType)
			: base(Opcode, Offset) {
			if (Opcode != OpCodes.Ret) {
				throw new ArgumentException();
			}
			m_ReturnType = ReturnType;
		}

		public override bool CanFallThrough {
			get {
				return false;
			}
		}

		public override int StackConsumeCount {
			get {
				System.Diagnostics.Debug.Assert(Opcode == OpCodes.Ret);
				System.Diagnostics.Debug.Assert(Opcode.StackBehaviourPop == StackBehaviour.Varpop);

				return (m_ReturnType == typeof(void)) ? 0 : 1;
			}
		}

		public override void WriteCode(System.IO.TextWriter writer, int indent, int CurStack) {
			WriteInstHeader(writer, indent);

			string ReturnArgument;
			if (StackConsumeCount == 0) {
				ReturnArgument = string.Empty;
			} else {
				ReturnArgument = " (" + StackName(CurStack) + ")";
			}
			writer.WriteLine("{0}return{1};", IndentString(indent), ReturnArgument);
		}

		public override List<Ever.HighLevel.Instruction> GetHighLevel(Ever.HighLevel.HlGraph Context) {
			List<HighLevel.Instruction> List = new List<Ever.HighLevel.Instruction>();
			HighLevel.Node Argument;

			if (StackConsumeCount == 0) {
				Argument = null;
			} else {
				Argument = Context.ReadStackLocationNode(Context.StackPointer);
			}

			List.Add(new HighLevel.ReturnInstruction(Argument));
			return List;
		}
	}
}
