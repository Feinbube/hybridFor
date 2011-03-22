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

        public BasicBlock(string LabelName)
        {
            labelName = LabelName;
        }

        public StackState EntryStackState { get; set; }
        public StackState ExitStackState { get; set; }

        public string LabelName { get { return labelName; } }

        public bool LabelNameUsed { get; set; }

        public List<Instruction> Instructions { get { return instructions; } }
        public List<BasicBlock> Successors { get { return successors; } }
    }
}
