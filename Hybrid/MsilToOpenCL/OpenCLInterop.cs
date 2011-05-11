﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Reflection;

namespace Hybrid.MsilToOpenCL
{
    internal class OpenCLInterop
    {
        internal static void CallOpenCLNet(int[] WorkSize, HlGraphEntry CacheEntry, InvokeContext ctx, HighLevel.HlGraph HLgraph, OpenCLNet.Device device)
        {
            // We can invoke the kernel using the arguments from ctx now :)
            if (device == null)
                device = GetFirstGpu();

            OpenCLNet.Platform Platform = device.Platform;

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
                    context = CacheEntry.Context = Platform.CreateContext(properties, new OpenCLNet.Device[] { device }, null, IntPtr.Zero);
                }

                program = CacheEntry.Program;
                if (program == null)
                {
                    program = context.CreateProgramWithSource(GetOpenCLSourceHeader(Platform, device) + CacheEntry.Source);

                    try
                    {
                        program.Build();
                    }
                    catch (Exception ex)
                    {
                        string err = program.GetBuildLog(device);
                        throw new Exception(err, ex);
                    }

                    CacheEntry.Program = program;
                }
            }

            using (CallContext CallContext = new CallContext(context, device, OpenCLNet.CommandQueueProperties.PROFILING_ENABLE, program.CreateKernel(HLgraph.MethodName)))
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

        internal static OpenCLNet.Device GetFirstGpu()
        {
            foreach (OpenCLNet.Platform platform in OpenCLNet.OpenCL.GetPlatforms())
                foreach (OpenCLNet.Device device in platform.QueryDevices(OpenCLNet.DeviceType.GPU))
                    return device;

            return null;
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

        internal static void WriteOpenCL(Type StructType, TextWriter writer)
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

        internal static void WriteOpenCL(HighLevel.HlGraph HLgraph, TextWriter writer)
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
