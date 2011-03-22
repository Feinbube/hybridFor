using System;
using System.Collections.Generic;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
	public enum InstructionType {
		Assignment,
		Return,
		Branch,
		ConditionalBranch
	}

	public abstract class Instruction {
		private InstructionType m_InstructionType;
		private Node m_Result;
		private Node m_Argument;

		public Instruction(InstructionType InstructionType)
			: this(InstructionType, null, null) {
		}

		public Instruction(InstructionType InstructionType, Node Result, Node Argument) {
			m_InstructionType = InstructionType;
			m_Result = Result;
			m_Argument = Argument;
		}

		public InstructionType InstructionType { get { return m_InstructionType; } }
		public Node Result { get { return m_Result; } set { m_Result = value; } }
		public Node Argument { get { return m_Argument; } set { m_Argument = value; } }
	}

	public class AssignmentInstruction : Instruction {
		public AssignmentInstruction(Node Result, Node Argument)
			: base(InstructionType.Assignment, Result, Argument) {
			if (Result != null && Result.DataType == null) {
				System.Diagnostics.Debug.Assert(Argument != null && Argument.DataType != null);
				Result.DataType = Argument.DataType;
			}
		}

		public override string ToString() {
			return ((Result == null) ? string.Empty : Result.ToString() + " = ") + (Argument == null ? "???;" : (Argument.ToString() + ";"));
		}
	}

	public class ReturnInstruction : Instruction {
		public ReturnInstruction(Node Argument)
			: base(InstructionType.Return, null, Argument) {
		}

		public override string ToString() {
			return (Argument == null) ? "return;" : "return (" + Argument.ToString() + ");";
		}
	}

	public class BranchInstruction : Instruction {
		private BasicBlock m_TargetBlock;

		public BranchInstruction(BasicBlock TargetBlock)
			: base(InstructionType.Branch) {
			m_TargetBlock = TargetBlock;
		}

		public BasicBlock TargetBlock { get { return m_TargetBlock; } set { m_TargetBlock = value; } }

		public override string ToString() {
			return (TargetBlock == null) ? "goto ???;" : ("goto " + TargetBlock.LabelName + ";");
		}
	}

	public class ConditionalBranchInstruction : Instruction {
		private BasicBlock m_TargetBlock;

		public ConditionalBranchInstruction(Node Argument, BasicBlock TargetBlock)
			: base(InstructionType.ConditionalBranch, null, Argument) {
			m_TargetBlock = TargetBlock;
		}

		public BasicBlock TargetBlock { get { return m_TargetBlock; } set { m_TargetBlock = value; } }

		public override string ToString() {
			string IfConstruct = (Argument == null) ? "if (???) " : "if (" + Argument.ToString() + ") ";
			return ((TargetBlock == null) ? (IfConstruct + "goto ???") : (IfConstruct + "goto " + TargetBlock.LabelName)) + ";";
		}
	}
}
