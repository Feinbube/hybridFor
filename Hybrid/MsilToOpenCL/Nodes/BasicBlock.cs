using System;
using System.Collections.Generic;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class BasicBlock
    {
        private string labelName;

        private List<Instruction> instructions = new List<Instruction>();
        private List<BasicBlock> successors = new List<BasicBlock>();

		private StackState m_EntryStackState;
		private StackState m_ExitStackState;
		private bool m_LabelNameUsed;

        public BasicBlock(string LabelName)
        {
            labelName = LabelName;
        }

		public StackState EntryStackState { get { return m_EntryStackState; } set { m_EntryStackState = value; } }
		public StackState ExitStackState { get { return m_ExitStackState; } set { m_ExitStackState = value; } }

        public string LabelName { get { return labelName; } }

		public bool LabelNameUsed { get { return m_LabelNameUsed; } set { m_LabelNameUsed = value; } }

        public List<Instruction> Instructions { get { return instructions; } }
        public List<BasicBlock> Successors { get { return successors; } }
    }
}
