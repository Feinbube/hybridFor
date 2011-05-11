#define USE_HOST_POINTER

using System;
using System.Threading;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Reflection;
using System.IO;

using OpenClKernel = System.Int32;

namespace Hybrid.MsilToOpenCL
{
    public class Parallel
    {
        private static HlGraphCache hlGraphCache = new HlGraphCache();
        private static int HlGraphSequenceNumber = 0;

        public static void ForGpu(int fromInclusive, int toExclusive, Action<int> action, OpenCLNet.Device device)
        {
            HlGraphEntry hlGraphEntry = GetHlGraph(action.Method, 1, device);

            System.Diagnostics.Debug.Assert(hlGraphEntry.fromInclusiveLocation != null && hlGraphEntry.fromInclusiveLocation.Count == 1);
            System.Diagnostics.Debug.Assert(hlGraphEntry.toExclusiveLocation != null && hlGraphEntry.toExclusiveLocation.Count == 1);

            using (InvokeContext ctx = new InvokeContext(hlGraphEntry.HlGraph))
            {
                if (hlGraphEntry.fromInclusiveLocation.Count > 0)
                    ctx.PutArgument(hlGraphEntry.fromInclusiveLocation[0], fromInclusive);

                if (hlGraphEntry.toExclusiveLocation.Count > 0)
                    ctx.PutArgument(hlGraphEntry.toExclusiveLocation[0], toExclusive);

                DoInvoke(new int[] { toExclusive - fromInclusive }, action.Target, hlGraphEntry, ctx, device);
            }
        }

        public static void ForGpu(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action, OpenCLNet.Device device)
        {
            HlGraphEntry hlGraphEntry = GetHlGraph(action.Method, 2, device);

            System.Diagnostics.Debug.Assert(hlGraphEntry.fromInclusiveLocation != null && hlGraphEntry.fromInclusiveLocation.Count == 2);
            System.Diagnostics.Debug.Assert(hlGraphEntry.toExclusiveLocation != null && hlGraphEntry.toExclusiveLocation.Count == 2);

            using (InvokeContext ctx = new InvokeContext(hlGraphEntry.HlGraph))
            {
                if (hlGraphEntry.fromInclusiveLocation.Count > 0)
                    ctx.PutArgument(hlGraphEntry.fromInclusiveLocation[0], fromInclusiveX);

                if (hlGraphEntry.toExclusiveLocation.Count > 0)
                    ctx.PutArgument(hlGraphEntry.toExclusiveLocation[0], toExclusiveX);

                if (hlGraphEntry.fromInclusiveLocation.Count > 1)
                    ctx.PutArgument(hlGraphEntry.fromInclusiveLocation[1], fromInclusiveY);

                if (hlGraphEntry.toExclusiveLocation.Count > 1)
                    ctx.PutArgument(hlGraphEntry.toExclusiveLocation[1], toExclusiveY);

                DoInvoke(new int[] { toExclusiveX - fromInclusiveX, toExclusiveY - fromInclusiveY }, action.Target, hlGraphEntry, ctx, device);
            }
        }

        private static void SetArguments(InvokeContext ctx, object Target, HighLevel.AccessPathEntry PathEntry)
        {
            if (PathEntry.ArgumentLocation != null)
                ctx.PutArgument(PathEntry.ArgumentLocation, Target);

            if (PathEntry.SubEntries != null)
                foreach (KeyValuePair<FieldInfo, HighLevel.AccessPathEntry> Entry in PathEntry.SubEntries)
                    SetArguments(ctx, Entry.Key.GetValue(Target), Entry.Value);
        }

        private static void DoInvoke(int[] WorkSize, object Target, HlGraphEntry CacheEntry, InvokeContext ctx, OpenCLNet.Device device)
        {
            HighLevel.HlGraph HLgraph = CacheEntry.HlGraph;

            foreach (KeyValuePair<FieldInfo, HighLevel.ArgumentLocation> Entry in HLgraph.StaticFieldMap)
                ctx.PutArgument(Entry.Value, Entry.Key.GetValue(null));

            SetArguments(ctx, Target, HLgraph.RootPathEntry);
            
            /*
            foreach (KeyValuePair<FieldInfo, HighLevel.ArgumentLocation> Entry in HLgraph.ThisFieldMap)
            {
                ctx.PutArgument(Entry.Value, Entry.Key.GetValue(Target));
            }
            foreach (KeyValuePair<FieldInfo, Dictionary<FieldInfo, HighLevel.ArgumentLocation>> Entry in HLgraph.OuterThisFieldMap) {
                object RealThis = Entry.Key.GetValue(Target);
                foreach (KeyValuePair<FieldInfo, HighLevel.ArgumentLocation> SubEntry in Entry.Value) {
                    ctx.PutArgument(SubEntry.Value, SubEntry.Key.GetValue(RealThis));
                }
            }*/

            foreach (KeyValuePair<HighLevel.ArgumentLocation, HighLevel.ArrayInfo> Entry in HLgraph.MultiDimensionalArrayInfo)
            {
                System.Diagnostics.Debug.Assert(Entry.Key.Index >= 0 && Entry.Key.Index < ctx.Arguments.Count);
                InvokeArgument BaseArrayArg = ctx.Arguments[Entry.Key.Index];
                System.Diagnostics.Debug.Assert(BaseArrayArg != null && BaseArrayArg.Value != null && BaseArrayArg.Value.GetType() == Entry.Key.DataType);
                System.Diagnostics.Debug.Assert(Entry.Key.DataType.IsArray && Entry.Key.DataType.GetArrayRank() == Entry.Value.DimensionCount);
                System.Diagnostics.Debug.Assert(BaseArrayArg.Value is Array);

                Array BaseArray = (System.Array)BaseArrayArg.Value;
                long BaseFactor = 1;
                for (int Dimension = 1; Dimension < Entry.Value.DimensionCount; Dimension++)
                {
                    int ThisDimensionLength = BaseArray.GetLength(Entry.Value.DimensionCount - 1 - (Dimension - 1));
                    BaseFactor *= ThisDimensionLength;
                    ctx.PutArgument(Entry.Value.ScaleArgument[Dimension], (int)BaseFactor);
                }
            }
            ctx.Complete();

            OpenCLInterop.CallOpenCLNet(WorkSize, CacheEntry, ctx, HLgraph, device);
        }

        public static int DumpCode = 0;	// 0-2: nothing, 3 = final, 4 = initial, 5 = after optimize, 6 = after OpenCL transform

        public static void PurgeCaches()
        {
            hlGraphCache.purge();
        }

        private static HlGraphEntry GetHlGraph(MethodInfo Method, int GidParamCount, OpenCLNet.Device device)
        {
            HlGraphEntry CacheEntry;
            HighLevel.HlGraph HLgraph;
            string MethodName;
            TextWriter writer = System.Console.Out;

            if (device == null)
                device = OpenCLInterop.GetFirstGpu();

            if(hlGraphCache.TryGetValue(device.DeviceID, Method, out CacheEntry))
                return CacheEntry;

            MethodName = string.Format("Cil2OpenCL_Root_Seq{0}", HlGraphSequenceNumber++);
            HLgraph = new HighLevel.HlGraph(Method, MethodName);

            if (DumpCode > 3)
                WriteCode(HLgraph, writer);

            // Optimize it (just some copy propagation and dead assignment elimination to get rid of
            // CIL stack accesses)
            HLgraph.Optimize();

            if (DumpCode > 4)
                WriteCode(HLgraph, writer);

            // Convert all expression trees into something OpenCL can understand
            HLgraph.ConvertForOpenCl();
            System.Diagnostics.Debug.Assert(!HLgraph.HasThisParameter);

            // Change the real first arguments (the "int"s of the Action<> method) to local variables
            // which get their value from OpenCL's built-in get_global_id() routine.
            // NOTE: ConvertArgumentToLocal removes the specified argument, so both calls need to specify
            //       an ArgumentId of zero!!!
            List<HighLevel.LocalVariableLocation> IdLocation = new List<HighLevel.LocalVariableLocation>();
            for (int i = 0; i < GidParamCount; i++)
            {
                IdLocation.Add(HLgraph.ConvertArgumentToLocal(0));
            }

            // Add fromInclusive and toExclusive as additional parameters
            List<HighLevel.ArgumentLocation> StartIdLocation = new List<HighLevel.ArgumentLocation>();
            List<HighLevel.ArgumentLocation> EndIdLocation = new List<HighLevel.ArgumentLocation>();
            for (int i = 0; i < GidParamCount; i++)
            {
                StartIdLocation.Add(HLgraph.InsertArgument(i * 2 + 0, "fromInclusive" + i, typeof(int), false));
                EndIdLocation.Add(HLgraph.InsertArgument(i * 2 + 1, "toExclusive" + i, typeof(int), false));
            }

            // "i0 = get_global_id(0) + fromInclusive0;"
            for (int i = 0; i < GidParamCount; i++)
            {
                HLgraph.CanonicalStartBlock.Instructions.Insert(i, new HighLevel.AssignmentInstruction(
                    new HighLevel.LocationNode(IdLocation[i]),
                    new HighLevel.AddNode(
                        new HighLevel.CallNode(typeof(OpenClFunctions).GetMethod("get_global_id", new Type[] { typeof(uint) }), new HighLevel.IntegerConstantNode(i)),
                        new HighLevel.LocationNode(StartIdLocation[i])
                        )
                    )
                );
            }

            // "if (i0 >= toExclusive0) return;"
            HighLevel.BasicBlock ReturnBlock = null;
            foreach (HighLevel.BasicBlock BB in HLgraph.BasicBlocks)
            {
                if (BB.Instructions.Count == 1 && BB.Instructions[0].InstructionType == HighLevel.InstructionType.Return)
                {
                    ReturnBlock = BB;
                    break;
                }
            }
            if (ReturnBlock == null)
            {
                ReturnBlock = new HighLevel.BasicBlock("CANONICAL_RETURN_BLOCK");
                ReturnBlock.Instructions.Add(new HighLevel.ReturnInstruction(null));
                HLgraph.BasicBlocks.Add(ReturnBlock);
            }
            ReturnBlock.LabelNameUsed = true;
            for (int i = 0; i < GidParamCount; i++)
            {
                HLgraph.CanonicalStartBlock.Instructions.Insert(GidParamCount + i, new HighLevel.ConditionalBranchInstruction(
                    new HighLevel.GreaterEqualsNode(
                        new HighLevel.LocationNode(IdLocation[i]),
                        new HighLevel.LocationNode(EndIdLocation[i])
                    ),
                    ReturnBlock
                    )
                );
            }

            if (DumpCode > 5)
            {
                WriteCode(HLgraph, writer);
            }

            // Update location usage information
            HLgraph.AnalyzeLocationUsage();

            // Finally, add the graph to the cache
            CacheEntry = new HlGraphEntry(HLgraph, StartIdLocation, EndIdLocation);

            // Get OpenCL source code
            CacheEntry.Source = getOpenCLSource(HLgraph);
            CacheEntry.Device = device;

            hlGraphCache.SetValue(device.DeviceID, Method, CacheEntry);

            return CacheEntry;
        }

        private static string getOpenCLSource(HighLevel.HlGraph HLgraph)
        {
            using (StringWriter Srcwriter = new StringWriter())
            {
                OpenCLInterop.WriteOpenCL(HLgraph, Srcwriter);
                string OpenClSource = Srcwriter.ToString();

                if (DumpCode > 2)
                    System.Console.WriteLine(OpenClSource);

                return OpenClSource;
            }
        }

        private static void WriteCode(HighLevel.HlGraph HLgraph, TextWriter writer)
        {
            writer.WriteLine("// begin {0}", HLgraph.MethodBase);

            if (HLgraph.MethodBase.IsConstructor)
            {
                writer.Write("constructor {0}::{1} (", ((System.Reflection.ConstructorInfo)HLgraph.MethodBase).DeclaringType, HLgraph.MethodBase.Name);
            }
            else
            {
                writer.Write("{0} {1}(", ((MethodInfo)HLgraph.MethodBase).ReturnType, HLgraph.MethodBase.Name);
            }

            for (int i = 0; i < HLgraph.Arguments.Count; i++)
            {
                if (i > 0)
                {
                    writer.Write(", ");
                }

                HighLevel.ArgumentLocation Argument = HLgraph.Arguments[i];
                string AttributeString = string.Empty;
                if ((Argument.Flags & HighLevel.LocationFlags.IndirectRead) != 0)
                {
                    AttributeString += "__deref_read ";
                }
                if ((Argument.Flags & HighLevel.LocationFlags.IndirectWrite) != 0)
                {
                    AttributeString += "__deref_write ";
                }

                writer.Write("{0}{1} {2}", AttributeString, Argument.DataType, Argument.Name);
            }

            writer.WriteLine(") {");

            foreach (HighLevel.LocalVariableLocation LocalVariable in HLgraph.LocalVariables)
            {
                writer.WriteLine("\t{0} {1};", LocalVariable.DataType, LocalVariable.Name);
            }

            for (int i = 0; i < HLgraph.BasicBlocks.Count; i++)
            {
                HighLevel.BasicBlock BB = HLgraph.BasicBlocks[i];

                if (BB == HLgraph.CanonicalEntryBlock || BB == HLgraph.CanonicalExitBlock)
                {
                    continue;
                }

                writer.WriteLine();
                writer.WriteLine("{0}:", BB.LabelName);
                foreach (HighLevel.Instruction Instruction in BB.Instructions)
                {
                    writer.WriteLine("\t{0}", Instruction.ToString());
                }

                if (BB.Successors.Count == 0)
                {
                    writer.WriteLine("\t// unreachable code");
                }
                else if (i + 1 == HLgraph.BasicBlocks.Count || HLgraph.BasicBlocks[i + 1] != BB.Successors[0])
                {
                    if (BB.Successors[0] == HLgraph.CanonicalExitBlock)
                    {
                        writer.WriteLine("\t// to canonical routine exit");
                    }
                    else
                    {
                        writer.WriteLine("\tgoto {0};", BB.Successors[0].LabelName);
                    }
                }
            }

            writer.WriteLine("}");
            writer.WriteLine("// end");
            writer.WriteLine();
        }
    }
}
