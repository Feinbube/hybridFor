﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public partial class HlGraph
    {
        private List<ArgumentLocation> m_Arguments = new List<ArgumentLocation>();
        public List<ArgumentLocation> Arguments { get { return m_Arguments; } }

        private List<LocalVariableLocation> m_LocalVariables = new List<LocalVariableLocation>();
        public List<LocalVariableLocation> LocalVariables { get { return m_LocalVariables; } }

        private List<StackLocation> m_CurrentStackInfo = new List<StackLocation>();

        private Dictionary<int, StackLocation> m_NewStackLocations = new Dictionary<int, StackLocation>();

        private int m_StackPointer;
        public int StackPointer { get { return m_StackPointer; } }

        private BasicBlock m_CanonicalEntryBlock;
        public BasicBlock CanonicalEntryBlock { get { return m_CanonicalEntryBlock; } }

        private BasicBlock m_CanonicalStartBlock;
        public BasicBlock CanonicalStartBlock { get { return m_CanonicalStartBlock; } }

        private BasicBlock m_CanonicalExitBlock;
        public BasicBlock CanonicalExitBlock { get { return m_CanonicalExitBlock; } }

        private List<BasicBlock> m_BasicBlockList = new List<BasicBlock>();
        public List<BasicBlock> BasicBlocks { get { return m_BasicBlockList; } }

        private Dictionary<int, BasicBlock> m_BasicBlockOffsetMap = new Dictionary<int, BasicBlock>();

        private Dictionary<System.Reflection.MethodInfo, HlGraphEntry> m_RelatedGraphs = new Dictionary<System.Reflection.MethodInfo, HlGraphEntry>();
        internal Dictionary<System.Reflection.MethodInfo, HlGraphEntry> RelatedGraphs { get { return m_RelatedGraphs; } }

        private int m_MaxStack;

        private bool m_IsKernel;
        public bool IsKernel { get { return m_IsKernel; } set { m_IsKernel = value; } }

        private System.Reflection.MethodBase m_MethodBase;
        public System.Reflection.MethodBase MethodBase { get { return m_MethodBase; } }

        private string m_MethodName;
        public string MethodName { get { return m_MethodName; } }

        public HlGraph(System.Reflection.MethodBase MethodBase, string MethodName)
        {
            m_MethodBase = MethodBase;
            m_MethodName = MethodName;

            ILgraph CilGraph = ILgraph.FromILbyteArray(MethodBase);

            // Initialize list of local variables
            System.Reflection.MethodBody MethodBody = MethodBase.GetMethodBody();

            IList<System.Reflection.LocalVariableInfo> LocalVariableInfos = MethodBody.LocalVariables;
            for (int i = 0; i < LocalVariableInfos.Count; i++)
            {
                System.Reflection.LocalVariableInfo Info = LocalVariableInfos[i];

                while (m_LocalVariables.Count <= Info.LocalIndex)
                {
                    m_LocalVariables.Add(null);
                }
                System.Diagnostics.Debug.Assert(m_LocalVariables[Info.LocalIndex] == null);
                m_LocalVariables[Info.LocalIndex] = new LocalVariableLocation(i, "local_" + Info.LocalIndex, Info.LocalType);
            }

            m_MaxStack = MethodBody.MaxStackSize;

            // Initialize list of arguments
            System.Reflection.ParameterInfo[] ParameterInfos = MethodBase.GetParameters();
            if ((MethodBase.CallingConvention & System.Reflection.CallingConventions.HasThis) != 0)
            {
                Type ThisArgumentType = MethodBase.DeclaringType;
                if (ThisArgumentType.IsValueType)
                {
                    ThisArgumentType = ThisArgumentType.Assembly.GetType(ThisArgumentType.FullName + "&", true);
                }
                CreateArgument("this", ThisArgumentType, true);
                m_HasThisParameter = true;
            }

            foreach (System.Reflection.ParameterInfo Info in ParameterInfos)
            {
                CreateArgument(Info.Name, Info.ParameterType, true);
            }

            // Initialize control flow graph
            Dictionary<ILBB, BasicBlock> BasicBlockMap = new Dictionary<ILBB, BasicBlock>();

            foreach (ILBB BB in CilGraph.BBlist)
            {
                string LabelName;
                if (BB == CilGraph.BBentry)
                {
                    LabelName = "CANONICAL_ROUTINE_ENTRY";
                }
                else if (BB == CilGraph.BBexit)
                {
                    LabelName = "CANONICAL_ROUTINE_EXIT";
                }
                else if (BB == CilGraph.BBstart)
                {
                    LabelName = "CANONICAL_ROUTINE_START";
                }
                else
                {
                    LabelName = string.Format("CIL_OFFSET_{0:X8}", BB.offset);
                }

                BasicBlock BasicBlock = new BasicBlock(LabelName);
                BasicBlockMap.Add(BB, BasicBlock);
                m_BasicBlockOffsetMap[BB.offset] = BasicBlock;
                m_BasicBlockList.Add(BasicBlock);

                if (BB.stackCountOnEntry.HasValue)
                {
                    BasicBlock.EntryStackState = new StackState(BB.stackCountOnEntry.Value, null);
                }
                if (BB.stackCountOnExit.HasValue)
                {
                    BasicBlock.ExitStackState = new StackState(BB.stackCountOnExit.Value, false);
                }
            }

            m_CanonicalEntryBlock = BasicBlockMap[CilGraph.BBentry];
            m_CanonicalStartBlock = BasicBlockMap[CilGraph.BBstart];
            m_CanonicalExitBlock = BasicBlockMap[CilGraph.BBexit];

            foreach (ILBB BB in CilGraph.BBlist)
            {
                BasicBlock BasicBlock = BasicBlockMap[BB];

                if (BB.FallThroughTarget != null)
                {
                    BasicBlock.Successors.Add(BasicBlockMap[BB.FallThroughTarget]);
                }

                if (BB.FinalTransfer != null)
                {
                    foreach (int Offset in BB.FinalTransfer.BranchTargetOffsets)
                    {
                        System.Diagnostics.Debug.Assert(m_BasicBlockOffsetMap.ContainsKey(Offset));
                        BasicBlock.Successors.Add(m_BasicBlockOffsetMap[Offset]);
                    }
                }
            }

            bool Changed;
            do
            {
                Changed = false;
                for (int i = 0; i < m_BasicBlockList.Count; i++)
                {
                    BasicBlock BasicBlock = m_BasicBlockList[i];

                    if (BasicBlock.EntryStackState != null && BasicBlock.EntryStackState.Complete && !BasicBlock.ExitStackState.Complete)
                    {
                        ILBB BB = CilGraph.BBlist[i];
                        System.Diagnostics.Debug.Assert(BasicBlockMap[BB] == BasicBlock);

                        m_CurrentStackInfo.Clear();
                        m_CurrentStackInfo.AddRange(BasicBlock.EntryStackState.StackLocations);
                        m_StackPointer = BasicBlock.EntryStackState.StackLocations.Count;

                        foreach (CilInstruction inst in BB.list)
                        {
                            m_NewStackLocations.Clear();

                            int OldStackPointer = m_StackPointer;
                            int StackProduce = inst.StackProduceCount;
                            int StackConsume = inst.StackConsumeCount;

                            List<Instruction> NewList = inst.GetHighLevel(this);
                            BasicBlock.Instructions.AddRange(NewList);

                            System.Diagnostics.Debug.Assert(m_CurrentStackInfo.Count >= StackConsume);
                            m_CurrentStackInfo.RemoveRange(m_CurrentStackInfo.Count - StackConsume, StackConsume);

                            for (int s = 0, TargetIndex = OldStackPointer - StackConsume; s < StackProduce; s++, TargetIndex++)
                            {
                                System.Diagnostics.Debug.Assert(m_NewStackLocations.ContainsKey(TargetIndex));
                                StackLocation NewStackLocation = m_NewStackLocations[TargetIndex];
                                m_NewStackLocations.Remove(TargetIndex);
                                m_CurrentStackInfo.Add(NewStackLocation);
                            }

                            System.Diagnostics.Debug.Assert(m_NewStackLocations.Count == 0);

                            m_StackPointer = OldStackPointer - StackConsume + StackProduce;
                        }

                        System.Diagnostics.Debug.Assert(m_StackPointer == BB.stackCountOnExit);
                        System.Diagnostics.Debug.Assert(m_CurrentStackInfo.Count == m_StackPointer);

                        BasicBlock.ExitStackState.StackLocations.Clear();
                        BasicBlock.ExitStackState.StackLocations.AddRange(m_CurrentStackInfo);
                        BasicBlock.ExitStackState.Complete = true;

                        // Merge this stack state onto successor block's entry state...
                        foreach (BasicBlock SuccessorBlock in BasicBlock.Successors)
                        {
                            Changed |= MergeStackInformation(BasicBlock.ExitStackState, SuccessorBlock.EntryStackState);
                        }
                    }
                }
            } while (Changed);

            // Check which label names are used by goto statements
            foreach (BasicBlock BB in BasicBlocks)
            {
                BB.LabelNameUsed = false;
            }

            for (int i = 2; i < BasicBlocks.Count; i++)
            {
                BasicBlock BB = BasicBlocks[i];

                if (BB.Successors.Count > 1)
                {
                    for (int k = 1; k < BB.Successors.Count; k++)
                    {
                        BB.Successors[k].LabelNameUsed = true;
                    }
                }

                if (BB.Successors.Count > 0 && (i + 1 == BasicBlocks.Count || BasicBlocks[i + 1] != BB.Successors[0]))
                {
                    BB.Successors[0].LabelNameUsed = true;
                }
            }

            // Cleanup
            foreach (BasicBlock BB in BasicBlocks)
            {
                if (BB.Instructions.Count > 0 && BB.Instructions[BB.Instructions.Count - 1].InstructionType == InstructionType.Branch)
                {
                    System.Diagnostics.Debug.Assert(BB.Successors.Count > 0 && BB.Successors[0] == ((BranchInstruction)BB.Instructions[BB.Instructions.Count - 1]).TargetBlock);
                    BB.Instructions.RemoveAt(BB.Instructions.Count - 1);
                }
            }
        }

        private bool MergeStackInformation(StackState SourceState, StackState TargetState)
        {
            System.Diagnostics.Debug.Assert(SourceState.Complete);
            if (SourceState.StackLocations.Count != TargetState.StackLocations.Count)
            {
                throw new InvalidOperationException("StackState.Merge: inconsistent number of stack locations");
            }

            bool Changed = false;
            for (int i = 0; i < SourceState.StackLocations.Count; i++)
            {
                if (TargetState.StackLocations[i].DataType == null)
                {
                    TargetState.StackLocations[i].DataType = SourceState.StackLocations[i].DataType;
                    Changed = true;
                }
                else if (TargetState.StackLocations[i].DataType != SourceState.StackLocations[i].DataType)
                {
                    // TODO: check whether these type objects are interoperable...
                    throw new InvalidOperationException(string.Format("StackState.Merge: source type '{0}' cannot be merged onto target type '{1}'.", SourceState.StackLocations[i].DataType, TargetState.StackLocations[i].DataType));
                }
            }

            if (!TargetState.Complete)
            {
                TargetState.Complete = true;
                Changed = true;
            }

            return Changed;
        }


        public StackLocation ReadStackLocation(int StackPointer)
        {
            int Index = StackPointer - 1;
            if (Index < 0 || Index >= m_CurrentStackInfo.Count)
            {
                throw new ArgumentOutOfRangeException("HighLevel.Context.StackLocation: StackPointer is out of range.");
            }
            return m_CurrentStackInfo[Index];
        }

        public LocationNode ReadStackLocationNode(int StackPointer)
        {
            return new LocationNode(ReadStackLocation(StackPointer));
        }

        public Node DefineStackLocationNode(int StackPointer)
        {
            return DefineStackLocationNode(StackPointer, null);
        }

        public Node DefineStackLocationNode(int StackPointer, Type DataType)
        {
            int Index = StackPointer - 1;
            System.Diagnostics.Debug.Assert(!m_NewStackLocations.ContainsKey(Index));

            StackLocation NewStackLocation = new StackLocation(Index, DataType);
            m_NewStackLocations[Index] = NewStackLocation;

            return new LocationNode(NewStackLocation);
        }

        public LocationNode ArgumentNode(int Index)
        {
            if (Index < 0 || Index >= m_Arguments.Count)
            {
                throw new ArgumentOutOfRangeException(string.Format("Argument index {0} is out of range.", Index));
            }

            return new LocationNode(m_Arguments[Index]);
        }

        public LocationNode LocalVariableNode(int Index)
        {
            if (Index < 0 || Index >= m_LocalVariables.Count)
            {
                throw new ArgumentOutOfRangeException(string.Format("Local variable index {0} is out of range.", Index));
            }

            return new LocationNode(m_LocalVariables[Index]);
        }


        public BasicBlock GetBlock(int Offset)
        {
            BasicBlock Block;
            if (!m_BasicBlockOffsetMap.TryGetValue(Offset, out Block))
            {
                throw new ArgumentException(string.Format("No basic block found that starts at offset {0:X}.", Offset));
            }
            return Block;
        }

        public Location StaticFieldLocation(System.Reflection.FieldInfo FieldInfo)
        {
            return new StaticFieldLocation(FieldInfo);
        }

        public bool TranslateCallNode(ref Node Result, ref Node CallNode)
        {
            if (CallNode == null)
            {
                throw new ArgumentNullException("CallNode");
            }

            System.Reflection.MethodInfo MethodInfo = ((CallNode)CallNode).MethodInfo;
            if (MethodInfo.DeclaringType.IsArray)
            {
                Type ArrayType = MethodInfo.DeclaringType;

                //
                // Check for canonical array access operators.
                //

                // An instance method  "t  t[ <n-times ','> ]::Get(i_1, ..., i_n)"  is the read accessor
                if (((MethodInfo.CallingConvention & System.Reflection.CallingConventions.HasThis) != 0) &&
                    MethodInfo.Name == "Get" &&
                    (MethodInfo.GetParameters().Length == ArrayType.GetArrayRank()) &&
                    (MethodInfo.ReturnType == ArrayType.GetElementType()))
                {
                    ArrayAccessNode NewNode = new ArrayAccessNode(ArrayType);
                    NewNode.SubNodes.AddRange(CallNode.SubNodes);
                    CallNode = NewNode;
                    return true;
                }

                // An instance method  "void  t[ <n-times ','> ]::Set(i_1, ..., i_n, value)" is the write accessor
                if (((MethodInfo.CallingConvention & System.Reflection.CallingConventions.HasThis) != 0) &&
                    MethodInfo.Name == "Set" &&
                    (MethodInfo.GetParameters().Length == ArrayType.GetArrayRank() + 1) &&
                    (MethodInfo.GetParameters()[ArrayType.GetArrayRank()].ParameterType == ArrayType.GetElementType()) &&
                    (MethodInfo.ReturnType == typeof(void)))
                {
                    System.Diagnostics.Debug.Assert(Result == null);

                    Result = new ArrayAccessNode(ArrayType);
                    for (int i = 0; i < CallNode.SubNodes.Count - 1; i++)
                    {
                        Result.SubNodes.Add(CallNode.SubNodes[i]);
                    }
                    CallNode = CallNode.SubNodes[CallNode.SubNodes.Count - 1];
                    return true;
                }

                // An instance method  "t&  t[ <n-times ','> ]::Address(i_1, ..., i_n)"  is the address-of accessor
                if (((MethodInfo.CallingConvention & System.Reflection.CallingConventions.HasThis) != 0) &&
                    MethodInfo.Name == "Address" &&
                    (MethodInfo.GetParameters().Length == ArrayType.GetArrayRank()) &&
                    (MethodInfo.ReturnType == Type.GetType(ArrayType.GetElementType().FullName + "&")))
                {
                    ArrayAccessNode NewNode = new ArrayAccessNode(ArrayType);
                    NewNode.SubNodes.AddRange(CallNode.SubNodes);
                    CallNode = new AddressOfNode(NewNode);
                    return true;
                }
            }

            // No translation possible or needed
            return false;
        }

        private Dictionary<System.Reflection.FieldInfo, ArgumentLocation> m_StaticFieldMap = new Dictionary<System.Reflection.FieldInfo, ArgumentLocation>();
        private Dictionary<System.Reflection.FieldInfo, ArgumentLocation> m_ThisFieldMap = new Dictionary<System.Reflection.FieldInfo, ArgumentLocation>();
        //private Dictionary<System.Reflection.FieldInfo, Dictionary<System.Reflection.FieldInfo, ArgumentLocation>> m_OuterThisFieldMap = new Dictionary<System.Reflection.FieldInfo, Dictionary<System.Reflection.FieldInfo, ArgumentLocation>>();
        public AccessPathEntry RootPathEntry = new AccessPathEntry();
        private Dictionary<ArgumentLocation, ArrayInfo> m_MultiDimensionalArrayInfo = new Dictionary<ArgumentLocation, ArrayInfo>();
        private bool m_HasThisParameter;
        private bool m_KeepThis;

        public Dictionary<System.Reflection.FieldInfo, ArgumentLocation> StaticFieldMap
        {
            get
            {
                return m_StaticFieldMap;
            }
        }

        public Dictionary<ArgumentLocation, ArrayInfo> MultiDimensionalArrayInfo
        {
            get
            {
                return m_MultiDimensionalArrayInfo;
            }
        }

        public bool HasThisParameter
        {
            get
            {
                return m_HasThisParameter;
            }
        }

        public bool KeepThis
        {
            get
            {
                return m_KeepThis;
            }
            set
            {
                m_KeepThis = value;
            }
        }

        public void ConvertForOpenCl(HlGraphCache RelatedGraphCache)
        {
            foreach (BasicBlock BasicBlock in BasicBlocks)
            {
                foreach (Instruction Instruction in BasicBlock.Instructions)
                {
                    Node Node = Instruction.Argument;
                    if (ConvertForOpenCl(ref Node, false, RelatedGraphCache))
                    {
                        Instruction.Argument = Node;
                    }
                    Node = Instruction.Result;
                    if (ConvertForOpenCl(ref Node, true, RelatedGraphCache))
                    {
                        Instruction.Result = Node;
                    }
                }
            }

            if (m_HasThisParameter && !m_KeepThis)
            {
                System.Diagnostics.Debug.Assert(m_Arguments.Count > 0 && m_Arguments[0].FromIL && m_Arguments[0].Name == "this" && m_Arguments[0].DataType == m_MethodBase.DeclaringType);
                DestroyArgument(0);
                m_HasThisParameter = false;
            }

            AnalyzeLocationUsage();

            foreach (LocalVariableLocation Location in LocalVariables)
            {
                if (!Location.DataType.IsPrimitive && (((Location.Flags & LocationFlags.Read) != 0) || ((Location.Flags & LocationFlags.Write) != 0)))
                {
                    throw new InvalidOperationException(string.Format("Sorry, local variable \'{0}\' of type \'{1}\' cannot be mapped to OpenCL.", Location.Name, Location.DataType.ToString()));
                }
            }

            foreach (ArgumentLocation Argument in this.Arguments)
            {
                if (((Argument.Flags & LocationFlags.Write) != 0) && ((Argument.Flags & LocationFlags.ForcePointer) == 0))
                {
                    Type NewType = Type.GetType(Argument.DataType.FullName + "&", true);
                    Argument.DataType = NewType;
                    Argument.Flags |= LocationFlags.ForcePointer;

                    foreach (BasicBlock BasicBlock in BasicBlocks)
                    {
                        foreach (Instruction Instruction in BasicBlock.Instructions)
                        {
                            Node Node = Instruction.Argument;
                            if (ConvertLocationToPointer(ref Node, Argument))
                            {
                                Instruction.Argument = Node;
                            }
                            Node = Instruction.Result;
                            if (ConvertLocationToPointer(ref Node, Argument))
                            {
                                Instruction.Result = Node;
                            }
                        }
                    }
                }
            }

            return;
        }

        public LocalVariableLocation ConvertArgumentToLocal(int Index)
        {
            if (m_Arguments.Count <= Index)
            {
                throw new ArgumentOutOfRangeException();
            }

            ArgumentLocation Argument = m_Arguments[Index];
            LocalVariableLocation NewLocalVariable = new LocalVariableLocation(m_LocalVariables.Count, Argument.Name, Argument.DataType);
            m_LocalVariables.Add(NewLocalVariable);

            foreach (BasicBlock BasicBlock in BasicBlocks)
            {
                foreach (Instruction Instruction in BasicBlock.Instructions)
                {
                    Node Node = Instruction.Argument;
                    if (ConvertLocation(ref Node, Argument, NewLocalVariable))
                    {
                        Instruction.Argument = Node;
                    }
                    Node = Instruction.Result;
                    if (ConvertLocation(ref Node, Argument, NewLocalVariable))
                    {
                        Instruction.Result = Node;
                    }
                }
            }

            DestroyArgument(Index);
            return NewLocalVariable;
        }

        public ArgumentLocation CreateArgument(string Name, Type DataType, bool FromIL)
        {
            return InsertArgument(m_Arguments.Count, Name, DataType, FromIL);
        }

        public ArgumentLocation InsertArgument(int Index, string Name, Type DataType, bool FromIL)
        {
            bool FoundName = true;
            string OriginalName = Name;
            int? SequenceNumber = null;
            while (FoundName)
            {
                FoundName = false;
                foreach (ArgumentLocation Location in m_Arguments)
                {
                    if (Location.Name == Name)
                    {
                        FoundName = true;
                        break;
                    }
                }
                if (!FoundName)
                {
                    foreach (LocalVariableLocation Location in m_LocalVariables)
                    {
                        if (Location.Name == Name)
                        {
                            FoundName = true;
                            break;
                        }
                    }
                }
                if (FoundName)
                {
                    if (SequenceNumber.HasValue)
                    {
                        SequenceNumber++;
                    }
                    else
                    {
                        SequenceNumber = 1;
                    }
                    Name = OriginalName + "_" + SequenceNumber.Value.ToString();
                }
            }

            ArgumentLocation ArgumentLocation = new ArgumentLocation(Index, Name, DataType, FromIL);
            m_Arguments.Insert(Index, ArgumentLocation);

            for (int i = Index + 1; i < m_Arguments.Count; i++)
            {
                m_Arguments[i].Index++;
            }
            return ArgumentLocation;
        }

        private void DestroyArgument(int Index)
        {
            m_Arguments.RemoveAt(Index);
            for (int i = Index; i < m_Arguments.Count; i++)
            {
                m_Arguments[i].Index--;
            }
        }

        private AccessPathEntry TraverseFieldAccess(InstanceFieldNode Node, bool IsArgument, AccessPathEntry PathEntry)
        {
            System.Reflection.FieldInfo FieldInfo = Node.FieldInfo;

            if (Node.SubNodes[0].NodeType == NodeType.Location && object.Equals(((LocationNode)Node.SubNodes[0]).Location, Arguments[0]))
            {
                AccessPathEntry NextPathEntry;
                if (PathEntry.SubEntries == null)
                {
                    PathEntry.SubEntries = new Dictionary<System.Reflection.FieldInfo, AccessPathEntry>();
                    PathEntry = PathEntry.SubEntries[FieldInfo] = new AccessPathEntry();
                }
                else if (!PathEntry.SubEntries.TryGetValue(FieldInfo, out NextPathEntry))
                {
                    PathEntry = PathEntry.SubEntries[FieldInfo] = new AccessPathEntry();
                }
                else
                {
                    PathEntry = NextPathEntry;
                }

                if (IsArgument)
                {
                    if (PathEntry.ArgumentLocation == null)
                    {
                        PathEntry.ArgumentLocation = CreateArgument("this_" + FieldInfo.Name, FieldInfo.FieldType, false);
                    }
                }
                return PathEntry;
            }
            else if (Node.SubNodes[0].NodeType == NodeType.InstanceField)
            {
                PathEntry = TraverseFieldAccess((InstanceFieldNode)Node.SubNodes[0], false, PathEntry);

                AccessPathEntry NextPathEntry;
                if (PathEntry.SubEntries == null)
                {
                    PathEntry.SubEntries = new Dictionary<System.Reflection.FieldInfo, AccessPathEntry>();
                    PathEntry = PathEntry.SubEntries[FieldInfo] = new AccessPathEntry();
                }
                else if (!PathEntry.SubEntries.TryGetValue(FieldInfo, out NextPathEntry))
                {
                    PathEntry = PathEntry.SubEntries[FieldInfo] = new AccessPathEntry();
                }
                else
                {
                    PathEntry = NextPathEntry;
                }

                if (IsArgument)
                {
                    if (PathEntry.ArgumentLocation == null)
                    {
//                        PathEntry.ArgumentLocation = CreateArgument("this_" + FieldInfo.Name, FieldInfo.FieldType, false);
                        PathEntry.ArgumentLocation = CreateArgument("this_" + FieldInfo.Name, FieldInfo.FieldType, false);
                    }
                }
                return PathEntry;
            }
            else
            {
                return null;
            }
        }

        private bool ConvertForOpenCl(ref Node Node, bool IsDef, HlGraphCache RelatedGraphCache)
        {
            bool Changed = false;

            if (!object.ReferenceEquals(Node, null))
            {
                Node.HlGraph = this;

                if (Node.NodeType == NodeType.InstanceField)
                {
                    System.Reflection.FieldInfo FieldInfo = ((InstanceFieldNode)Node).FieldInfo;

                    bool UnsupportedFieldAccess = false;

                    if ((MethodBase.CallingConvention & System.Reflection.CallingConventions.HasThis) == 0)
                    {
                        UnsupportedFieldAccess = true;
                    }
                    else
                    {
                        AccessPathEntry PathEntry = TraverseFieldAccess((InstanceFieldNode)Node, true, RootPathEntry);

                        if (PathEntry == null)
                        {
                            UnsupportedFieldAccess = true;
                        }
                        else
                        {
                            System.Diagnostics.Debug.Assert(PathEntry.ArgumentLocation != null);
                            Node = new LocationNode(PathEntry.ArgumentLocation);
                            Changed = true;
                        }
                    }

                    if (UnsupportedFieldAccess)
                    {
                        throw new InvalidOperationException(string.Format("Field access operation '{0}' not supported.", Node));
                    }
                }
                else if (Node.NodeType == NodeType.Location)
                {
                    LocationNode LocationNode = (LocationNode)Node;
                    if (LocationNode.Location.LocationType == LocationType.CilStack)
                    {
                        throw new InvalidOperationException(string.Format("Reference to CIL stack '{0}' could not be eliminated.", Node));
                    }
                    else if (LocationNode.Location.LocationType == LocationType.StaticField)
                    {
                        System.Reflection.FieldInfo FieldInfo = ((StaticFieldLocation)LocationNode.Location).FieldInfo;

                        ArgumentLocation ArgumentLocation;
                        if (!StaticFieldMap.TryGetValue(FieldInfo, out ArgumentLocation))
                        {
                            ArgumentLocation = CreateArgument("static_" + FieldInfo.Name, FieldInfo.FieldType, false);
                            StaticFieldMap[FieldInfo] = ArgumentLocation;
                        }
                        Node = new LocationNode(ArgumentLocation);
                        Changed = true;
                    }
                    else if (m_HasThisParameter && !m_KeepThis && (LocationNode.Location.LocationType == LocationType.Argument && LocationNode.Location == m_Arguments[0]))
                    {
                        throw new InvalidOperationException(string.Format("Reference to 'this' parameter could not be eliminated."));
                    }
                }
                else
                {
                    for (int i = 0; i < Node.SubNodes.Count; i++)
                    {
                        Node SubNode = Node.SubNodes[i];
                        if (ConvertForOpenCl(ref SubNode, false, RelatedGraphCache))
                        {
                            Node.SubNodes[i] = SubNode;
                            Changed = true;
                        }
                    }

                    if (Node.NodeType == NodeType.ArrayAccess)
                    {
                        ArrayAccessNode ArrayAccessNode = (ArrayAccessNode)Node;
                        if (ArrayAccessNode.ArrayType.GetArrayRank() > 1)
                        {
                            if (ArrayAccessNode.SubNodes.Count == 0 || ArrayAccessNode.SubNodes[0].NodeType != NodeType.Location ||
                                ((LocationNode)ArrayAccessNode.SubNodes[0]).Location.LocationType != LocationType.Argument)
                            {
                                throw new InvalidOperationException(string.Format("Sorry, multi-dimensional array access '{0}' could not be resolved.", ArrayAccessNode));
                            }

                            ArgumentLocation Argument = (ArgumentLocation)((LocationNode)ArrayAccessNode.SubNodes[0]).Location;
                            ArrayInfo ArrayInfo;
                            if (!MultiDimensionalArrayInfo.TryGetValue(Argument, out ArrayInfo))
                            {
                                ArrayInfo = new ArrayInfo(Argument);

                                for (int Dimension = 0; Dimension < ArrayInfo.DimensionCount; Dimension++)
                                {
                                    Node ScaleNode;
                                    ArgumentLocation ScaleArgument;
                                    if (Dimension == 0)
                                    {
                                        ScaleArgument = null;
                                        ScaleNode = new IntegerConstantNode(1);
                                    }
                                    else
                                    {
                                        ScaleArgument = CreateArgument(Argument.Name + "_scdim_" + Dimension.ToString(), typeof(int), false);
                                        ScaleNode = new LocationNode(ScaleArgument);
                                    }
                                    ArrayInfo.ScaleNode[Dimension] = ScaleNode;
                                    ArrayInfo.ScaleArgument[Dimension] = ScaleArgument;
                                }

                                MultiDimensionalArrayInfo[Argument] = ArrayInfo;
                            }
                            if (ArrayAccessNode.SubNodes.Count != ArrayAccessNode.ArrayType.GetArrayRank() + 1)
                            {
                                throw new InvalidOperationException(string.Format("Sorry, multi-dimensional array access '{0}' specified invalid number of array dimensions.", ArrayAccessNode));
                            }

                            Node SingleDimensionalIndexNode = null;
                            for (int Dimension = 0; Dimension < ArrayInfo.DimensionCount; Dimension++)
                            {
                                Node DimensionIndexNode = ArrayAccessNode.SubNodes[ArrayAccessNode.SubNodes.Count - 1 - Dimension];
                                Node DimensionScaleNode = ArrayInfo.ScaleNode[Dimension];

                                if (Dimension == 0)
                                {
                                    SingleDimensionalIndexNode = DimensionIndexNode;
                                }
                                else
                                {
                                    SingleDimensionalIndexNode = new AddNode(new MulNode(DimensionScaleNode, DimensionIndexNode), SingleDimensionalIndexNode);
                                }
                            }

                            ArrayAccessNode.SubNodes.RemoveRange(2, ArrayAccessNode.SubNodes.Count - 2);
                            ArrayAccessNode.SubNodes[1] = SingleDimensionalIndexNode;
                            ArrayAccessNode.FlattenArrayType();
                        }
                    }
                    else if (Node.NodeType == NodeType.Call)
                    {
                        CallNode CallNode = (CallNode)Node;
                        if (string.IsNullOrEmpty(GetOpenClFunctionName(CallNode.MethodInfo)))
                        {
                            // Not a built-in method, so we need to check whether this method can be compiled from CIL...

                            HlGraphEntry RelatedGraphEntry;

                            if (!m_RelatedGraphs.TryGetValue(CallNode.MethodInfo, out RelatedGraphEntry) &&
                                !RelatedGraphCache.TryGetValue(IntPtr.Zero, CallNode.MethodInfo, out RelatedGraphEntry))
                            {
                                RelatedGraphEntry = Parallel.ConstructRelatedHlGraphEntry(CallNode.MethodInfo, this, RelatedGraphCache);
                            }

                            if (RelatedGraphEntry == null)
                            {
                                throw new InvalidOperationException(string.Format("Sorry, no equivalent OpenCL function call available for '{0}'.", CallNode));
                            }

                            // Map additional parameters introduced as part of OpenCL conversion of the RelatedGraphEntry
                            for (int i = CallNode.SubNodes.Count; i < RelatedGraphEntry.HlGraph.Arguments.Count; i++)
                            {
                                ArgumentLocation RelatedArgument = RelatedGraphEntry.HlGraph.Arguments[i];

                                // Static fields
                                foreach (KeyValuePair<System.Reflection.FieldInfo, ArgumentLocation> RelatedMapEntry in RelatedGraphEntry.HlGraph.StaticFieldMap)
                                {
                                    if (object.ReferenceEquals(RelatedMapEntry.Value, RelatedArgument))
                                    {
                                        System.Reflection.FieldInfo FieldInfo = RelatedMapEntry.Key;

                                        ArgumentLocation ArgumentLocation;
                                        if (!StaticFieldMap.TryGetValue(FieldInfo, out ArgumentLocation))
                                        {
                                            ArgumentLocation = CreateArgument("static_" + FieldInfo.Name, FieldInfo.FieldType, false);
                                            StaticFieldMap[FieldInfo] = ArgumentLocation;
                                        }
                                        CallNode.SubNodes.Add(new LocationNode(ArgumentLocation));
                                        RelatedArgument = null;
                                        break;
                                    }
                                }

                                if (object.ReferenceEquals(RelatedArgument, null))
                                    continue;

                                // TODO: multi-dimensional arrays, references to "this", ...

                                if (!object.ReferenceEquals(RelatedArgument, null))
                                {
                                    // Not good. This code won't compile...
                                    throw new InvalidOperationException(string.Format("Unable to map additional argument '{0}' (index {1}) in related function call '{2}'.",
                                        RelatedArgument, i + 1, CallNode.ToString()));
                                }
                            }
                        }
                    }
                }
            }

            return Changed;
        }

        private bool ConvertLocationToPointer(ref Node Node, Location Location)
        {
            bool Changed = false;

            if (!object.ReferenceEquals(Node, null))
            {
                if (Node.NodeType == NodeType.Location)
                {
                    LocationNode LocationNode = (LocationNode)Node;
                    if (LocationNode.Location == Location)
                    {
                        Node = new DerefNode(LocationNode);
                        Changed = true;
                    }
                }
                else if (Node.NodeType == NodeType.AddressOf)
                {
                    System.Diagnostics.Debug.Assert(Node.SubNodes.Count == 1);
                    if ((Node.SubNodes[0].NodeType == NodeType.Location) && ((LocationNode)Node.SubNodes[0]).Location == Location)
                    {
                        Node = Node.SubNodes[0];
                        Changed = true;
                    }
                }
                else
                {
                    for (int i = 0; i < Node.SubNodes.Count; i++)
                    {
                        Node SubNode = Node.SubNodes[i];
                        if (ConvertLocationToPointer(ref SubNode, Location))
                        {
                            Node.SubNodes[i] = SubNode;
                            Changed = true;
                        }
                    }
                }
            }

            return Changed;
        }

        public string GetOpenClFunctionName(System.Reflection.MethodInfo MethodInfo)
        {
            //
            // Check for some hard-coded translations first
            //

            if (MethodInfo.DeclaringType == typeof(System.Math))
            {
                switch (MethodInfo.Name)
                {
                    case "Acos":
                    case "Asin":
                    case "Atan":
                    case "Atan2":
                    case "Cos":
                    case "Cosh":
                    case "Exp":
                    case "Log10":
                    case "Max":
                    case "Min":
                    case "Sin":
                    case "Sinh":
                    case "Sqrt":
                    case "Tan":
                    case "Tanh":
                        return MethodInfo.Name.ToLower();

                    case "Log":
                        if (MethodInfo.GetParameters().Length == 1)
                        {
                            return MethodInfo.Name.ToLower();
                        }
                        break;	// Logarithm to arbitrary base is not a built-in in OpenCL
                }

                // This function has not been mapped yet...
                return null;
            }

            // TODO for Random.Next() as well

            //
            // Check for an explicitly linked routine
            //

            HlGraphEntry RelatedGraphEntry;
            if (m_RelatedGraphs.TryGetValue(MethodInfo, out RelatedGraphEntry))
            {
                return object.ReferenceEquals(RelatedGraphEntry, null) ? null : RelatedGraphEntry.HlGraph.MethodName;
            }

            //
            // Finally, check for an explicit OpenClAlias name
            //

            return OpenClAliasAttribute.Get(MethodInfo);
        }

        public void AnalyzeLocationUsage()
        {
            foreach (BasicBlock BasicBlock in this.BasicBlocks)
            {
                foreach (Instruction Instruction in BasicBlock.Instructions)
                {
                    LocationUsage LocationUsage = LocationUsage.ForInstruction(Instruction, RelatedGraphs);
                    foreach (Location Location in LocationUsage.DefinedLocations)
                    {
                        Location.Flags |= LocationFlags.Write;
                    }
                    foreach (Location Location in LocationUsage.IndirectDefinedLocations)
                    {
                        Location.Flags |= LocationFlags.IndirectWrite;
                    }
                    foreach (Location Location in LocationUsage.UsedLocations)
                    {
                        Location.Flags |= LocationFlags.Read;
                    }
                    foreach (Location Location in LocationUsage.IndirectUsedLocations)
                    {
                        Location.Flags |= LocationFlags.IndirectRead;
                    }
                }
            }
        }

        private bool ConvertLocation(ref Node Node, Location OldLocation, Location NewLocation)
        {
            bool Changed = false;

            if (!object.ReferenceEquals(Node, null))
            {
                if (Node.NodeType == NodeType.Location)
                {
                    LocationNode LocationNode = (LocationNode)Node;
                    if (LocationNode.Location == OldLocation)
                    {
                        LocationNode.Location = NewLocation;
                    }
                }
                else
                {
                    for (int i = 0; i < Node.SubNodes.Count; i++)
                    {
                        Node SubNode = Node.SubNodes[i];
                        if (ConvertLocation(ref SubNode, OldLocation, NewLocation))
                        {
                            Node.SubNodes[i] = SubNode;
                            Changed = true;
                        }
                    }
                }
            }

            return Changed;
        }
    }
}
