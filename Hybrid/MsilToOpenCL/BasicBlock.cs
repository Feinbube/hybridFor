using System;
using System.Collections.Generic;
using System.Text;

namespace Ever.HighLevel {
	public class BasicBlock {
		private string m_LabelName;
		private bool m_LabelNameUsed;
		private List<Instruction> m_Instructions = new List<Instruction>();
		private List<BasicBlock> m_Successors = new List<BasicBlock>();

		private StackState m_EntryStackState;
		private StackState m_ExitStackState;

		public BasicBlock(string LabelName) {
			m_LabelName = LabelName;
		}

		public StackState EntryStackState { get { return m_EntryStackState; } set { m_EntryStackState = value; } }
		public StackState ExitStackState { get { return m_ExitStackState; } set { m_ExitStackState = value; } }

		public string LabelName { get { return m_LabelName; } }
		public bool LabelNameUsed { get { return m_LabelNameUsed; } set { m_LabelNameUsed = value; } }
		public List<Instruction> Instructions { get { return m_Instructions; } }
		public List<BasicBlock> Successors { get { return m_Successors; } }
	}
}
