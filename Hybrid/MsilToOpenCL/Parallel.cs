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
        private static Dictionary<MethodInfo, HlGraphCacheEntry> HlGraphCache = new Dictionary<MethodInfo, HlGraphCacheEntry>();
        private static int HlGraphSequenceNumber = 0;

        public static void ForGpu(int fromInclusive, int toExclusive, Action<int> action)
        {
            HlGraphCacheEntry CacheEntry = GetHlGraph(action.Method, 1);

            System.Diagnostics.Debug.Assert(CacheEntry.fromInclusiveLocation != null && CacheEntry.fromInclusiveLocation.Count == 1);
            System.Diagnostics.Debug.Assert(CacheEntry.toExclusiveLocation != null && CacheEntry.toExclusiveLocation.Count == 1);

            using (InvokeContext ctx = new InvokeContext(CacheEntry.HlGraph))
            {
                if (CacheEntry.fromInclusiveLocation.Count > 0)
                    ctx.PutArgument(CacheEntry.fromInclusiveLocation[0], fromInclusive);

                if (CacheEntry.toExclusiveLocation.Count > 0)
                    ctx.PutArgument(CacheEntry.toExclusiveLocation[0], toExclusive);

                DoInvoke(new int[] { toExclusive - fromInclusive }, action.Target, CacheEntry, ctx);
            }
        }

        public static void ForGpu(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action)
        {
            HlGraphCacheEntry CacheEntry = GetHlGraph(action.Method, 2);

            System.Diagnostics.Debug.Assert(CacheEntry.fromInclusiveLocation != null && CacheEntry.fromInclusiveLocation.Count == 2);
            System.Diagnostics.Debug.Assert(CacheEntry.toExclusiveLocation != null && CacheEntry.toExclusiveLocation.Count == 2);

            using (InvokeContext ctx = new InvokeContext(CacheEntry.HlGraph))
            {
                if (CacheEntry.fromInclusiveLocation.Count > 0)
                    ctx.PutArgument(CacheEntry.fromInclusiveLocation[0], fromInclusiveX);

                if (CacheEntry.toExclusiveLocation.Count > 0)
                    ctx.PutArgument(CacheEntry.toExclusiveLocation[0], toExclusiveX);

                if (CacheEntry.fromInclusiveLocation.Count > 1)
                    ctx.PutArgument(CacheEntry.fromInclusiveLocation[1], fromInclusiveY);

                if (CacheEntry.toExclusiveLocation.Count > 1)
                    ctx.PutArgument(CacheEntry.toExclusiveLocation[1], toExclusiveY);

                DoInvoke(new int[] { toExclusiveX - fromInclusiveX, toExclusiveY - fromInclusiveY }, action.Target, CacheEntry, ctx);
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

        private static void DoInvoke(int[] WorkSize, object Target, HlGraphCacheEntry CacheEntry, InvokeContext ctx)
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

            callOpenCLNet(WorkSize, CacheEntry, ctx, HLgraph);
        }

        private static void callOpenCLNet(int[] WorkSize, HlGraphCacheEntry CacheEntry, InvokeContext ctx, HighLevel.HlGraph HLgraph)
        {
            // We can invoke the kernel using the arguments from ctx now :)
            OpenCLNet.Platform Platform = OpenCLNet.OpenCL.GetPlatform(0);
            OpenCLNet.Device[] Devices = Platform.QueryDevices(OpenCLNet.DeviceType.ALL);
            OpenCLNet.Device Device = Devices[0];

            OpenCLNet.Context context;
            OpenCLNet.Program program;

            lock (CacheEntry)
            {
                context = CacheEntry.Context;
                if (context == null)
                {
                    IntPtr[] properties = new IntPtr[]
                    {
                        new IntPtr((long)OpenCLNet.ContextProperties.PLATFORM), Platform.PlatformID,
                        IntPtr.Zero,
                    };
                    context = CacheEntry.Context = Platform.CreateContext(properties, new OpenCLNet.Device[] { Device }, null, IntPtr.Zero);
                }

                program = CacheEntry.Program;
                if (program == null)
                {
                    program = context.CreateProgramWithSource(GetOpenCLSourceHeader(Platform, Device) + CacheEntry.Source);

                    try
                    {
                        program.Build();
                    }
                    catch (Exception ex)
                    {
                        string err = program.GetBuildLog(Device);
                        throw new Exception(err, ex);
                    }

                    CacheEntry.Program = program;
                }
            }

            using (CallContext CallContext = new CallContext(context, Device, OpenCLNet.CommandQueueProperties.PROFILING_ENABLE, program.CreateKernel(HLgraph.MethodName)))
            {
                OpenCLNet.CommandQueue commandQueue = CallContext.CommandQueue;

                for (int i = 0; i < ctx.Arguments.Count; i++)
                {
                    ctx.Arguments[i].WriteToKernel(CallContext, i);
                }

                OpenCLNet.Event StartEvent, EndEvent;

                commandQueue.EnqueueMarker(out StartEvent);

                IntPtr[] GlobalWorkSize = new IntPtr[WorkSize.Length];
                for (int i = 0; i < WorkSize.Length; i++)
                {
                    GlobalWorkSize[i] = new IntPtr(WorkSize[i]);
                }
                commandQueue.EnqueueNDRangeKernel(CallContext.Kernel, (uint)GlobalWorkSize.Length, null, GlobalWorkSize, null);

                for (int i = 0; i < ctx.Arguments.Count; i++)
                {
                    ctx.Arguments[i].ReadFromKernel(CallContext, i);
                }

                commandQueue.Finish();
                commandQueue.EnqueueMarker(out EndEvent);
                commandQueue.Finish();

                ulong StartTime, EndTime;
                StartEvent.GetEventProfilingInfo(OpenCLNet.ProfilingInfo.QUEUED, out StartTime);
                EndEvent.GetEventProfilingInfo(OpenCLNet.ProfilingInfo.END, out EndTime);
            }
        }

        public static int DumpCode = 0;	// 0-2: nothing, 3 = final, 4 = initial, 5 = after optimize, 6 = after OpenCL transform

        public static void PurgeCaches()
        {
            lock (HlGraphCache)
            {
                foreach (KeyValuePair<MethodInfo, HlGraphCacheEntry> Entry in HlGraphCache)
                {
                    if (Entry.Value != null)
                    {
                        Entry.Value.Dispose();
                    }
                }
                HlGraphCache.Clear();
            }
        }

        private static HlGraphCacheEntry GetHlGraph(MethodInfo Method, int GidParamCount)
        {
            HlGraphCacheEntry CacheEntry;
            HighLevel.HlGraph HLgraph;
            string MethodName;

            lock (HlGraphCache)
            {
                if (HlGraphCache.TryGetValue(Method, out CacheEntry))
                {
                    return CacheEntry;
                }
                MethodName = string.Format("Cil2OpenCL_Root_Seq{0}", HlGraphSequenceNumber++);
            }

            TextWriter writer = System.Console.Out;

            HLgraph = new HighLevel.HlGraph(Method, MethodName);

            if (DumpCode > 3)
            {
                WriteCode(HLgraph, writer);
            }

            // Optimize it (just some copy propagation and dead assignment elimination to get rid of
            // CIL stack accesses)
            HLgraph.Optimize();

            if (DumpCode > 4)
            {
                WriteCode(HLgraph, writer);
            }

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
            CacheEntry = new HlGraphCacheEntry(HLgraph, StartIdLocation, EndIdLocation);

            // Get OpenCL source code
            string OpenClSource;
            using (StringWriter Srcwriter = new StringWriter())
            {
                WriteOpenCL(HLgraph, Srcwriter);
                OpenClSource = Srcwriter.ToString();

                if (DumpCode > 2)
                {
                    System.Console.WriteLine(OpenClSource);
                }
            }
            CacheEntry.Source = OpenClSource;

            lock (HlGraphCache)
            {
                HlGraphCache[Method] = CacheEntry;
            }

            return CacheEntry;
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

        private static void WriteOpenCL(Type StructType, TextWriter writer)
        {
            if (StructType == null)
            {
                throw new ArgumentNullException("StructType");
            }
            else if (!StructType.IsValueType)
            {
                throw new ArgumentException(string.Format("Unable to generate OpenCL code for non-ValueType '{0}'", StructType.FullName));
            }

            writer.WriteLine("// OpenCL structure definition for type '{0}'", StructType.FullName);
            writer.WriteLine("struct {0} {{", StructType.Name);
            FieldInfo[] Fields = StructType.GetFields();
            foreach (FieldInfo Field in Fields)
            {
                writer.WriteLine("\t{0} {1};", GetOpenClType(Field.FieldType), Field.Name);
            }
            writer.WriteLine("}}");
            writer.WriteLine();
        }

        private static string GetOpenCLSourceHeader(OpenCLNet.Platform Platform, OpenCLNet.Device Device)
        {
            System.Text.StringBuilder String = new System.Text.StringBuilder();

            String.AppendLine("// BEGIN GENERATED OpenCL");

            if (Device.HasExtension("cl_amd_fp64"))
            {
                String.AppendLine("#pragma OPENCL EXTENSION cl_amd_fp64 : enable");
            }
            else if (Device.HasExtension("cl_khr_fp64"))
            {
                String.AppendLine("#pragma OPENCL EXTENSION cl_khr_fp64 : enable");
            }

            if (Device.HasExtension("cl_khr_global_int32_base_atomics"))
            {
                String.AppendLine("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable");
            }

            String.AppendLine();

            return String.ToString();
        }

        private static void WriteOpenCL(HighLevel.HlGraph HLgraph, TextWriter writer)
        {
            writer.WriteLine("// OpenCL kernel for method '{0}' of type '{1}'", HLgraph.MethodBase.ToString(), HLgraph.MethodBase.DeclaringType.ToString());
            writer.WriteLine("__kernel {0} {1}(", GetOpenClType(((MethodInfo)HLgraph.MethodBase).ReturnType), HLgraph.MethodName);
            for (int i = 0; i < HLgraph.Arguments.Count; i++)
            {
                HighLevel.ArgumentLocation Argument = HLgraph.Arguments[i];

                string AttributeString = string.Empty;
                if ((Argument.Flags & HighLevel.LocationFlags.IndirectRead) != 0)
                {
                    AttributeString += "/*[in";
                }
                if ((Argument.Flags & HighLevel.LocationFlags.IndirectWrite) != 0)
                {
                    if (AttributeString == string.Empty)
                    {
                        AttributeString += "/*[out";
                    }
                    else
                    {
                        AttributeString += ",out";
                    }
                }

                if (AttributeString != string.Empty) { AttributeString += "]*/ "; }

                if (Argument.DataType.IsArray || Argument.DataType.IsPointer)
                {
                    AttributeString += "__global ";
                }

                writer.WriteLine("\t{0}{1} {2}{3}", AttributeString, GetOpenClType(Argument.DataType), Argument.Name, i + 1 < HLgraph.Arguments.Count ? "," : string.Empty);
            }
            writer.WriteLine(")");
            writer.WriteLine("/*");
            writer.WriteLine("  Generated by CIL2OpenCL");
            writer.WriteLine("*/");
            writer.WriteLine("{");

            foreach (HighLevel.LocalVariableLocation LocalVariable in HLgraph.LocalVariables)
            {
                string AttributeString = string.Empty;
                if ((LocalVariable.Flags & HighLevel.LocationFlags.Read) != 0)
                {
                    if (AttributeString == string.Empty) { AttributeString += "/*["; } else { AttributeString += ","; }
                    AttributeString += "read";
                }
                if ((LocalVariable.Flags & HighLevel.LocationFlags.Write) != 0)
                {
                    if (AttributeString == string.Empty) { AttributeString += "/*["; } else { AttributeString += ","; }
                    AttributeString += "write";
                }
                if ((LocalVariable.Flags & HighLevel.LocationFlags.IndirectRead) != 0)
                {
                    if (AttributeString == string.Empty) { AttributeString += "/*["; } else { AttributeString += ","; }
                    AttributeString += "deref_read";
                }
                if ((LocalVariable.Flags & HighLevel.LocationFlags.IndirectWrite) != 0)
                {
                    if (AttributeString == string.Empty) { AttributeString += "/*["; } else { AttributeString += ","; }
                    AttributeString += "deref_write";
                }
                if (AttributeString == string.Empty) { AttributeString = "/*UNUSED*/ // "; } else { AttributeString += "]*/ "; }

                writer.WriteLine("\t{0}{1} {2};", AttributeString, GetOpenClType(LocalVariable.DataType), LocalVariable.Name);
            }

            HighLevel.BasicBlock FallThroughTargetBlock = HLgraph.CanonicalStartBlock;
            for (int i = 0; i < HLgraph.BasicBlocks.Count; i++)
            {
                HighLevel.BasicBlock BB = HLgraph.BasicBlocks[i];

                if (BB == HLgraph.CanonicalEntryBlock || BB == HLgraph.CanonicalExitBlock)
                {
                    continue;
                }

                if (FallThroughTargetBlock != null && FallThroughTargetBlock != BB)
                {
                    writer.WriteLine("\tgoto {0};", FallThroughTargetBlock.LabelName);
                }

                FallThroughTargetBlock = null;

                writer.WriteLine();
                if (BB.LabelNameUsed)
                {
                    writer.WriteLine("{0}:", BB.LabelName);
                }
                else
                {
                    writer.WriteLine("//{0}: (unreferenced block label)", BB.LabelName);
                }

                foreach (HighLevel.Instruction Instruction in BB.Instructions)
                {
                    writer.WriteLine("\t{0}", Instruction.ToString());
                }

                if (BB.Successors.Count == 0)
                {
                    writer.WriteLine("\t// End of block is unreachable");
                }
                else if (BB.Successors[0] == HLgraph.CanonicalExitBlock)
                {
                    writer.WriteLine("\t// End of block is unreachable/canonical routine exit");
                }
                else
                {
                    FallThroughTargetBlock = BB.Successors[0];
                }
            }

            writer.WriteLine("}");
            writer.WriteLine("// END GENERATED OpenCL");
        }

        public static string GetOpenClType(Type DataType)
        {
            return InnerGetOpenClType(DataType);
        }

        private static string InnerGetOpenClType(Type DataType)
        {
            if (DataType == typeof(void))
            {
                return "void";
            }
            else if (DataType == typeof(sbyte))
            {
                return "char";
            }
            else if (DataType == typeof(byte))
            {
                return "uchar";
            }
            else if (DataType == typeof(short))
            {
                return "short";
            }
            else if (DataType == typeof(ushort))
            {
                return "ushort";
            }
            else if (DataType == typeof(int) || DataType == typeof(IntPtr) || DataType == typeof(bool))
            {
                return "int";
            }
            else if (DataType == typeof(uint) || DataType == typeof(UIntPtr))
            {
                return "uint";
            }
            else if (DataType == typeof(long))
            {
                return "long";
            }
            else if (DataType == typeof(ulong))
            {
                return "ulong";
            }
            else if (DataType == typeof(float))
            {
                return "float";
            }
            else if (DataType == typeof(double))
            {
                return "double";
            }
            else if (DataType.IsArray)
            {
                return InnerGetOpenClType(DataType.GetElementType()) + "*";
            }
            else
            {
                throw new ArgumentException(string.Format("Sorry, data type '{0}' cannot be mapped to OpenCL.", DataType));
            }
        }
    }
}
