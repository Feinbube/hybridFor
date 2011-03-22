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
	/// <summary>
	/// This attribute marks methods that represent built-in OpenCL functions.
	/// If the Alias property is not specified, the .NET name of the method is used as-is in
	/// OpenCL. Otherwise, if the Alias is present, its value is used instead of the internal
	/// .NET name.
	/// </summary>
	[AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
	public sealed class OpenClAliasAttribute : Attribute {
		private string m_Alias;
		public OpenClAliasAttribute()
			: this(string.Empty) {
		}

		public OpenClAliasAttribute(string Alias) {
			m_Alias = Alias;
		}

		public string Alias { get { return m_Alias; } }

		public static string Get(MethodInfo MethodInfo) {
			if (MethodInfo == null) {
				throw new ArgumentNullException("MethodInfo");
			}

			object[] Alias = MethodInfo.GetCustomAttributes(typeof(OpenClAliasAttribute), false);
			if (Alias.Length == 0) {
				return null;
			} else if (string.IsNullOrEmpty(((OpenClAliasAttribute)Alias[0]).Alias)) {
				return MethodInfo.Name;
			} else {
				return ((OpenClAliasAttribute)Alias[0]).Alias;
			}
		}
	}

	[AttributeUsage(AttributeTargets.Parameter)]
	public class OpenClIn : Attribute {
	}

	[AttributeUsage(AttributeTargets.Parameter)]
	public class OpenClOut : Attribute {
	}

	public static class OpenClFunctions {
		public static void __target_only() {
			string Name;
			try {
				System.Diagnostics.StackTrace StackTrace = new System.Diagnostics.StackTrace(Thread.CurrentThread, false);
				System.Diagnostics.StackFrame StackFrame = StackTrace.GetFrame(1);
				Name = StackFrame.GetMethod().Name;
			} catch (System.Exception ex) {
				throw new InvalidOperationException("Sorry, the target function does not have an implementation on the host CPU and can only be invoked from OpenCL.", ex);
			}
			throw new InvalidOperationException(string.Format("Sorry, the function '{0}' does not have an implementation on the host CPU and can only be invoked from OpenCL.", Name));
		}

		[OpenClAlias]
		public static uint get_work_dim() {
			__target_only();
			return 0;
		}

		[OpenClAlias]
		public static uint get_global_size(uint dimindx) {
			__target_only();
			return 0;
		}

		[OpenClAlias]
		public static uint get_global_id(uint dimindx) {
			__target_only();
			return 0;
		}

		[OpenClAlias]
		public static uint get_local_size(uint dimindx) {
			__target_only();
			return 0;
		}

		[OpenClAlias]
		public static uint get_local_id(uint dimindx) {
			__target_only();
			return 0;
		}

		[OpenClAlias]
		public static uint get_num_groups(uint dimindx) {
			__target_only();
			return 0;
		}

		[OpenClAlias]
		public static uint get_group_id(uint dimindx) {
			__target_only();
			return 0;
		}
	}

	public class Parallel {
		#region Singleton

		Parallel instance = null;

		public Parallel Instance {
			get {
				if (instance == null) {
					instance = new Parallel();
					instance.Initialize();
				}

				return instance;
			}
		}

		private Parallel() { }

		#endregion

		private void Initialize() {
		}

		private class CallContext : IDisposable {
			public OpenCLNet.Context Context;
			public OpenCLNet.CommandQueue CommandQueue;
			public OpenCLNet.Kernel Kernel;

			public CallContext(OpenCLNet.Context Context, OpenCLNet.Device Device, OpenCLNet.CommandQueueProperties CqProperties, OpenCLNet.Kernel Kernel) {
				this.Context = Context;
				this.CommandQueue = Context.CreateCommandQueue(Device, CqProperties);
				this.Kernel = Kernel;
			}

			~CallContext() {
				Dispose();
			}

			public void Dispose() {
				if (Kernel != null) {
					Kernel.Dispose();
					Kernel = null;
				}
				if (CommandQueue != null) {
					CommandQueue.Dispose();
					CommandQueue = null;
				}
				System.GC.SuppressFinalize(this);
			}
		}

		private abstract class InvokeArgument : IDisposable {
			public abstract object Value { get; }

			public override string ToString() {
				object Value = this.Value;
				return (object.ReferenceEquals(Value, null)) ? "<null>" : Value.ToString();
			}

			public abstract void WriteToKernel(CallContext CallContext, int i);
			public abstract void ReadFromKernel(CallContext CallContext, int i);

			#region IDisposable Members

			public virtual void Dispose() {
			}

			#endregion

			public class Int32Arg : InvokeArgument {
				private int m_Value;

				public Int32Arg(int Value) {
					m_Value = Value;
				}

				public override object Value { get { return m_Value; } }
				public override string ToString() { return m_Value.ToString(); }

				public override void WriteToKernel(CallContext CallContext, int i) {
					CallContext.Kernel.SetArg(i, m_Value);
				}

				public override void ReadFromKernel(CallContext CallContext, int i) {
					// Nothing to do
				}
			}

			public class UInt32Arg : InvokeArgument {
				private uint m_Value;

				public UInt32Arg(uint Value) {
					m_Value = Value;
				}

				public override object Value { get { return m_Value; } }
				public override string ToString() { return m_Value.ToString(); }

				public override void WriteToKernel(CallContext CallContext, int i) {
					CallContext.Kernel.SetArg(i, m_Value);
				}

				public override void ReadFromKernel(CallContext CallContext, int i) {
					// Nothing to do
				}
			}

			public class Int64Arg : InvokeArgument {
				private long m_Value;

				public Int64Arg(long Value) {
					m_Value = Value;
				}

				public override object Value { get { return m_Value; } }
				public override string ToString() { return m_Value.ToString(); }

				public override void WriteToKernel(CallContext CallContext, int i) {
					CallContext.Kernel.SetArg(i, m_Value);
				}

				public override void ReadFromKernel(CallContext CallContext, int i) {
					// Nothing to do
				}
			}

			public class UInt64Arg : InvokeArgument {
				private ulong m_Value;

				public UInt64Arg(ulong Value) {
					m_Value = Value;
				}

				public override object Value { get { return m_Value; } }
				public override string ToString() { return m_Value.ToString(); }

				public override void WriteToKernel(CallContext CallContext, int i) {
					CallContext.Kernel.SetArg(i, m_Value);
				}

				public override void ReadFromKernel(CallContext CallContext, int i) {
					// Nothing to do
				}
			}

			public class FloatArg : InvokeArgument {
				private float m_Value;

				public FloatArg(float Value) {
					m_Value = Value;
				}

				public override object Value { get { return m_Value; } }
				public override string ToString() { return m_Value.ToString(); }

				public override void WriteToKernel(CallContext CallContext, int i) {
					CallContext.Kernel.SetArg(i, m_Value);
				}

				public override void ReadFromKernel(CallContext CallContext, int i) {
					// Nothing to do
				}
			}

			public class DoubleArg : InvokeArgument {
				private double m_Value;

				public DoubleArg(double Value) {
					m_Value = Value;
				}

				public override object Value { get { return m_Value; } }
				public override string ToString() { return m_Value.ToString(); }

				public override void WriteToKernel(CallContext CallContext, int i) {
					CallContext.Kernel.SetArg(i, m_Value);
				}

				public override void ReadFromKernel(CallContext CallContext, int i) {
					// Nothing to do
				}
			}

			public class ArrayArg : InvokeArgument {
				private System.Array m_Value;
				private OpenCLNet.Mem MemBuffer;
				private bool m_ForRead;
				private bool m_ForWrite;
				private System.Runtime.InteropServices.GCHandle? m_GCHandle;

				public ArrayArg(System.Array Value, bool ForRead, bool ForWrite) {
					if (!ForRead && !ForWrite) {
						ForRead = ForWrite = true;
					}

					m_Value = Value;
					m_ForRead = ForRead;
					m_ForWrite = ForWrite;
				}

				public override object Value { get { return m_Value; } }
				public override string ToString() { return m_Value.ToString(); }

				public override void WriteToKernel(CallContext CallContext, int i) {
					//
					// Get count and size of individual array elements
					//

					long ElementCount = 1;
					for (int d = 0; d < m_Value.Rank; d++) {
						ElementCount *= (m_Value.GetUpperBound(d) - m_Value.GetLowerBound(d) + 1);
					}

					Type RealElementType = Value.GetType().GetElementType();
					int ElementSize = System.Runtime.InteropServices.Marshal.SizeOf(RealElementType);

					//
					// Allocate memory buffer on target hardware using the appropriate type
					//

					OpenCLNet.MemFlags MemFlags;
					if (m_ForRead && !m_ForWrite) {
						MemFlags = OpenCLNet.MemFlags.READ_ONLY;
					} else if (!m_ForRead && m_ForWrite) {
						MemFlags = OpenCLNet.MemFlags.WRITE_ONLY;
					} else {
                        m_ForRead = m_ForWrite = true;
						MemFlags = OpenCLNet.MemFlags.READ_WRITE;
					}

#if USE_HOST_POINTER
					m_GCHandle = System.Runtime.InteropServices.GCHandle.Alloc(Value, System.Runtime.InteropServices.GCHandleType.Pinned);

					MemBuffer = CallContext.Context.CreateBuffer(MemFlags | OpenCLNet.MemFlags.USE_HOST_PTR, ElementCount * ElementSize, m_GCHandle.Value.AddrOfPinnedObject());
					CallContext.Kernel.SetArg(i, MemBuffer);
#else
					MemBuffer = CallContext.Context.CreateBuffer(MemFlags, ElementCount * ElementSize, IntPtr.Zero);
					CallContext.Kernel.SetArg(i, MemBuffer);

					//
					// If the buffer is read by the device, transfer the data
					//

					if (m_ForRead) {
						System.Runtime.InteropServices.GCHandle gch = System.Runtime.InteropServices.GCHandle.Alloc(Value, System.Runtime.InteropServices.GCHandleType.Pinned);

						try {
							IntPtr p = gch.Value.AddrOfPinnedObject();
							CallContext.CommandQueue.EnqueueWriteBuffer(MemBuffer, true, IntPtr.Zero, new IntPtr(ElementCount * ElementSize), p);
						} finally {
							gch.Free();
						}
					}
#endif
				}

				private static void ConvertDoubleToFloatArray(Array DoubleArray, Array FloatArray, int[] dims, int[] widx, int[] ridx, int p) {
					if (p == 0) {
						int LowDim = DoubleArray.GetLowerBound(0);
						for (int i = 0; i < dims[0]; i++) {
							widx[0] = i;
							ridx[0] = i + LowDim;
							FloatArray.SetValue((float)(double)DoubleArray.GetValue(ridx), widx);
						}
					} else {
						int LowDim = DoubleArray.GetLowerBound(p);
						for (int i = 0; i < dims[p]; i++) {
							widx[p] = i;
							ridx[p] = i + LowDim;
							ConvertDoubleToFloatArray(DoubleArray, FloatArray, dims, widx, ridx, p - 1);
						}
					}
				}

				public override void ReadFromKernel(CallContext CallContext, int i) {
					//
					// Do nothing if the device is guaranteed not to write to this buffer
					//

					if (!m_ForWrite) {
						return;
					}

					//
					// Get count and size of individual array elements
					//

					long ElementCount = 1;
					for (int d = 0; d < m_Value.Rank; d++) {
						ElementCount *= (m_Value.GetUpperBound(d) - m_Value.GetLowerBound(d) + 1);
					}

					Type RealElementType = Value.GetType().GetElementType();
					int ElementSize = System.Runtime.InteropServices.Marshal.SizeOf(RealElementType);

					//
					// Read buffer back into main memory
					//

					CallContext.CommandQueue.EnqueueReadBuffer(MemBuffer, true, IntPtr.Zero, new IntPtr(ElementCount * ElementSize), m_GCHandle.Value.AddrOfPinnedObject());
				}

				#region IDisposable Members

				public override void Dispose() {
					base.Dispose();
					if (MemBuffer != null) {
						MemBuffer.Dispose();
						MemBuffer = null;
					}
					if (m_GCHandle.HasValue) {
						m_GCHandle.Value.Free();
						m_GCHandle = null;
					}
				}

				#endregion
			}
		}

		private class InvokeContext : IDisposable {
			private List<InvokeArgument> m_Arguments;
			public List<InvokeArgument> Arguments {
				get {
					return m_Arguments;
				}
			}

			public InvokeContext(HighLevel.HlGraph HLgraph) {
				m_Arguments = new List<InvokeArgument>(HLgraph.Arguments.Count);
				for (int i = 0; i < HLgraph.Arguments.Count; i++) {
					System.Diagnostics.Debug.Assert(HLgraph.Arguments[i].Index == i);
					m_Arguments.Add(null);
				}
			}

			public void PutArgument(HighLevel.ArgumentLocation Location, object Value) {
				int Index = Location.Index;

				System.Diagnostics.Debug.Assert(Index >= 0 && Index < m_Arguments.Count);
				System.Diagnostics.Debug.Assert(m_Arguments[Index] == null);

				InvokeArgument Argument = null;
				if (object.ReferenceEquals(Value, null)) {
					Argument = new InvokeArgument.Int32Arg(0);
				} else if (Value is int) {
					Argument = new InvokeArgument.Int32Arg((int)Value);
				} else if (Value is uint) {
					Argument = new InvokeArgument.UInt32Arg((uint)Value);
				} else if (Value is long) {
					Argument = new InvokeArgument.Int64Arg((long)Value);
				} else if (Value is ulong) {
					Argument = new InvokeArgument.UInt64Arg((ulong)Value);
				} else if (Value is float) {
					Argument = new InvokeArgument.FloatArg((float)Value);
				} else if (Value is double) {
					Argument = new InvokeArgument.DoubleArg((double)Value);
				} else {
					Type Type = Value.GetType();
					if (Type.IsArray) {
						Type ElementType = Type.GetElementType();
						if (ElementType == typeof(byte) || ElementType == typeof(int) || ElementType == typeof(uint) || ElementType == typeof(long) || ElementType == typeof(ulong)
							|| ElementType == typeof(float) || ElementType == typeof(double)) {
							bool ForRead = false, ForWrite = false;
							if ((Location.Flags & HighLevel.LocationFlags.IndirectRead) != 0) { ForRead = true; }
							if ((Location.Flags & HighLevel.LocationFlags.IndirectWrite) != 0) { ForWrite = true; }

							if (!ForRead && !ForWrite) {
								ForRead = ForWrite = true;
							}

							Argument = new InvokeArgument.ArrayArg((System.Array)Value, ForRead, ForWrite);
						}
					}
				}

				if (Argument == null) {
					throw new InvalidOperationException(string.Format("Sorry, argument type '{0}' cannot be marshalled for OpenCL.", Value.GetType()));
				}

				m_Arguments[Index] = Argument;
			}

			public void Complete() {
				int Index = m_Arguments.IndexOf(null);
				System.Diagnostics.Debug.Assert(Index < 0);
				if (Index >= 0) {
					throw new InvalidOperationException(string.Format("Argument {0} is not assigned.", Index));
				}
			}

			#region IDisposable Members

			public void Dispose() {
				foreach (InvokeArgument Argument in m_Arguments) {
					if (Argument != null) {
						Argument.Dispose();
					}
				}
				m_Arguments.Clear();
				System.GC.SuppressFinalize(this);
			}

			#endregion
		}

		internal class HlGraphCacheEntry : IDisposable {
			private HighLevel.HlGraph m_HlGraph;
			private List<HighLevel.ArgumentLocation> m_fromInclusiveLocation;
			private List<HighLevel.ArgumentLocation> m_toExclusiveLocation;
			private string m_Source;

			public HlGraphCacheEntry(HighLevel.HlGraph HlGraph, List<HighLevel.ArgumentLocation> fromInclusiveLocation, List<HighLevel.ArgumentLocation> toExclusiveLocation) {
				m_HlGraph = HlGraph;
				m_fromInclusiveLocation = fromInclusiveLocation;
				m_toExclusiveLocation = toExclusiveLocation;
			}

			public HighLevel.HlGraph HlGraph { get { return m_HlGraph; } }
			public List<HighLevel.ArgumentLocation> fromInclusiveLocation { get { return m_fromInclusiveLocation; } }
			public List<HighLevel.ArgumentLocation> toExclusiveLocation { get { return m_toExclusiveLocation; } }
			public string Source { get { return m_Source; } set { m_Source = value; } }

			private OpenCLNet.Context m_Context;
			private OpenCLNet.Program m_Program;
			public OpenCLNet.Context Context { get { return m_Context; } set { m_Context = value; } }
			public OpenCLNet.Program Program { get { return m_Program; } set { m_Program = value; } }

            ~HlGraphCacheEntry() {
                Dispose();
            }

            public void Dispose() {
                if (m_Program != null) {
                    m_Program.Dispose();
                    m_Program = null;
                }
                if (m_Context != null) {
                    m_Context.Dispose();
                    m_Context = null;
                }
                System.GC.SuppressFinalize(this);
            }
        }

		private static Dictionary<MethodInfo, HlGraphCacheEntry> HlGraphCache = new Dictionary<MethodInfo, HlGraphCacheEntry>();
		private static int HlGraphSequenceNumber = 0;

		public static void ForGpu(int fromInclusive, int toExclusive, Action<int> action) {
			HlGraphCacheEntry CacheEntry = GetHlGraph(action.Method, 1);

			System.Diagnostics.Debug.Assert(CacheEntry.fromInclusiveLocation != null && CacheEntry.fromInclusiveLocation.Count == 1);
			System.Diagnostics.Debug.Assert(CacheEntry.toExclusiveLocation != null && CacheEntry.toExclusiveLocation.Count == 1);

			using (InvokeContext ctx = new InvokeContext(CacheEntry.HlGraph)) {
				if (CacheEntry.fromInclusiveLocation.Count > 0) {
					ctx.PutArgument(CacheEntry.fromInclusiveLocation[0], fromInclusive);
				}
				if (CacheEntry.toExclusiveLocation.Count > 0) {
					ctx.PutArgument(CacheEntry.toExclusiveLocation[0], toExclusive);
				}

				DoInvoke(new int[] { toExclusive - fromInclusive }, action.Target, CacheEntry, ctx);
			}
		}

		public static void ForGpu(int fromInclusiveX, int toExclusiveX, int fromInclusiveY, int toExclusiveY, Action<int, int> action) {
			HlGraphCacheEntry CacheEntry = GetHlGraph(action.Method, 2);

			System.Diagnostics.Debug.Assert(CacheEntry.fromInclusiveLocation != null && CacheEntry.fromInclusiveLocation.Count == 2);
			System.Diagnostics.Debug.Assert(CacheEntry.toExclusiveLocation != null && CacheEntry.toExclusiveLocation.Count == 2);

			using (InvokeContext ctx = new InvokeContext(CacheEntry.HlGraph)) {
				if (CacheEntry.fromInclusiveLocation.Count > 0) {
					ctx.PutArgument(CacheEntry.fromInclusiveLocation[0], fromInclusiveX);
				}
				if (CacheEntry.toExclusiveLocation.Count > 0) {
					ctx.PutArgument(CacheEntry.toExclusiveLocation[0], toExclusiveX);
				}
				if (CacheEntry.fromInclusiveLocation.Count > 1) {
					ctx.PutArgument(CacheEntry.fromInclusiveLocation[1], fromInclusiveY);
				}
				if (CacheEntry.toExclusiveLocation.Count > 1) {
					ctx.PutArgument(CacheEntry.toExclusiveLocation[1], toExclusiveY);
				}

				DoInvoke(new int[] { toExclusiveX - fromInclusiveX, toExclusiveY - fromInclusiveY }, action.Target, CacheEntry, ctx);
			}
		}

		private static void SetArguments(InvokeContext ctx, object Target, HighLevel.AccessPathEntry PathEntry) {
			if (PathEntry.ArgumentLocation != null) {
				ctx.PutArgument(PathEntry.ArgumentLocation, Target);
			}
			if (PathEntry.SubEntries != null) {
				foreach (KeyValuePair<FieldInfo, HighLevel.AccessPathEntry> Entry in PathEntry.SubEntries) {
					SetArguments(ctx, Entry.Key.GetValue(Target), Entry.Value);
				}
			}
		}

		private static void DoInvoke(int[] WorkSize, object Target, HlGraphCacheEntry CacheEntry, InvokeContext ctx) {
			HighLevel.HlGraph HLgraph = CacheEntry.HlGraph;

			foreach (KeyValuePair<FieldInfo, HighLevel.ArgumentLocation> Entry in HLgraph.StaticFieldMap) {
				ctx.PutArgument(Entry.Value, Entry.Key.GetValue(null));
			}

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
			foreach (KeyValuePair<HighLevel.ArgumentLocation, HighLevel.ArrayInfo> Entry in HLgraph.MultiDimensionalArrayInfo) {
				System.Diagnostics.Debug.Assert(Entry.Key.Index >= 0 && Entry.Key.Index < ctx.Arguments.Count);
				InvokeArgument BaseArrayArg = ctx.Arguments[Entry.Key.Index];
				System.Diagnostics.Debug.Assert(BaseArrayArg != null && BaseArrayArg.Value != null && BaseArrayArg.Value.GetType() == Entry.Key.DataType);
				System.Diagnostics.Debug.Assert(Entry.Key.DataType.IsArray && Entry.Key.DataType.GetArrayRank() == Entry.Value.DimensionCount);
				System.Diagnostics.Debug.Assert(BaseArrayArg.Value is Array);

				Array BaseArray = (System.Array)BaseArrayArg.Value;
				long BaseFactor = 1;
				for (int Dimension = 1; Dimension < Entry.Value.DimensionCount; Dimension++) {
					int ThisDimensionLength = BaseArray.GetLength(Entry.Value.DimensionCount - 1 - (Dimension - 1));
					BaseFactor *= ThisDimensionLength;
					ctx.PutArgument(Entry.Value.ScaleArgument[Dimension], (int)BaseFactor);
				}
			}
			ctx.Complete();

			// We can invoke the kernel using the arguments from ctx now :)
			OpenCLNet.Platform Platform = OpenCLNet.OpenCL.GetPlatform(0);
			OpenCLNet.Device[] Devices = Platform.QueryDevices(OpenCLNet.DeviceType.ALL);
            OpenCLNet.Device Device = Devices[0];

			OpenCLNet.Context Context;
			OpenCLNet.Program Program;
			lock (CacheEntry) {
				Context = CacheEntry.Context;
                if (Context == null) {
                    IntPtr[] properties = new IntPtr[]{
                        new IntPtr((long)OpenCLNet.ContextProperties.PLATFORM), Platform.PlatformID,
                        IntPtr.Zero,
                    };
                    Context = CacheEntry.Context = Platform.CreateContext(properties, new OpenCLNet.Device[] { Device }, null, IntPtr.Zero);
                    //Context = CacheEntry.Context = Platform.CreateDefaultContext();
                }
				Program = CacheEntry.Program;
				if (Program == null) {
					Program = Context.CreateProgramWithSource(GetOpenCLSourceHeader(Platform, Device) + CacheEntry.Source);

					try {
						Program.Build();
					} catch (Exception ex) {
						string err = Program.GetBuildLog(Device);
						throw new Exception(err, ex);
					}

					CacheEntry.Program = Program;
				}
			}
            
			using (CallContext CallContext = new CallContext(Context, Device, OpenCLNet.CommandQueueProperties.PROFILING_ENABLE, Program.CreateKernel(HLgraph.MethodName))) {
				OpenCLNet.CommandQueue CQ = CallContext.CommandQueue;
                
				for (int i = 0; i < ctx.Arguments.Count; i++) {
					ctx.Arguments[i].WriteToKernel(CallContext, i);
				}

				OpenCLNet.Event StartEvent, EndEvent;

				CQ.EnqueueMarker(out StartEvent);

				IntPtr[] GlobalWorkSize = new IntPtr[WorkSize.Length];
				for (int i = 0; i < WorkSize.Length; i++) {
					GlobalWorkSize[i] = new IntPtr(WorkSize[i]);
				}
				CQ.EnqueueNDRangeKernel(CallContext.Kernel, (uint)GlobalWorkSize.Length, null, GlobalWorkSize, null);

				for (int i = 0; i < ctx.Arguments.Count; i++) {
					ctx.Arguments[i].ReadFromKernel(CallContext, i);
				}

				CQ.Finish();
				CQ.EnqueueMarker(out EndEvent);
				CQ.Finish();

				ulong StartTime, EndTime;
				StartEvent.GetEventProfilingInfo(OpenCLNet.ProfilingInfo.QUEUED, out StartTime);
				EndEvent.GetEventProfilingInfo(OpenCLNet.ProfilingInfo.END, out EndTime);
			}
		}

		public static int DumpCode = 0;	// 0-2: nothing, 3 = final, 4 = initial, 5 = after optimize, 6 = after OpenCL transform

        public static void PurgeCaches() {
            lock (HlGraphCache) {
                foreach (KeyValuePair<MethodInfo, HlGraphCacheEntry> Entry in HlGraphCache) {
                    if (Entry.Value != null) {
                        Entry.Value.Dispose();
                    }
                }
                HlGraphCache.Clear();
            }
        }

		private static HlGraphCacheEntry GetHlGraph(MethodInfo Method, int GidParamCount) {
			HlGraphCacheEntry CacheEntry;
			HighLevel.HlGraph HLgraph;
			string MethodName;

			lock (HlGraphCache) {
				if (HlGraphCache.TryGetValue(Method, out CacheEntry)) {
					return CacheEntry;
				}
				MethodName = string.Format("Cil2OpenCL_Root_Seq{0}", HlGraphSequenceNumber++);
			}

			TextWriter writer = System.Console.Out;

			HLgraph = new HighLevel.HlGraph(Method, MethodName);

			if (DumpCode > 3) {
				WriteCode(HLgraph, writer);
			}

			// Optimize it (just some copy propagation and dead assignment elimination to get rid of
			// CIL stack accesses)
			HLgraph.Optimize();

			if (DumpCode > 4) {
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
			for (int i = 0; i < GidParamCount; i++) {
				IdLocation.Add(HLgraph.ConvertArgumentToLocal(0));
			}

			// Add fromInclusive and toExclusive as additional parameters
			List<HighLevel.ArgumentLocation> StartIdLocation = new List<HighLevel.ArgumentLocation>();
			List<HighLevel.ArgumentLocation> EndIdLocation = new List<HighLevel.ArgumentLocation>();
			for (int i = 0; i < GidParamCount; i++) {
				StartIdLocation.Add(HLgraph.InsertArgument(i * 2 + 0, "fromInclusive" + i, typeof(int), false));
				EndIdLocation.Add(HLgraph.InsertArgument(i * 2 + 1, "toExclusive" + i, typeof(int), false));
			}

			// "i0 = get_global_id(0) + fromInclusive0;"
			for (int i = 0; i < GidParamCount; i++) {
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
			foreach (HighLevel.BasicBlock BB in HLgraph.BasicBlocks) {
				if (BB.Instructions.Count == 1 && BB.Instructions[0].InstructionType == HighLevel.InstructionType.Return) {
					ReturnBlock = BB;
					break;
				}
			}
			if (ReturnBlock == null) {
				ReturnBlock = new HighLevel.BasicBlock("CANONICAL_RETURN_BLOCK");
				ReturnBlock.Instructions.Add(new HighLevel.ReturnInstruction(null));
				HLgraph.BasicBlocks.Add(ReturnBlock);
			}
			ReturnBlock.LabelNameUsed = true;
			for (int i = 0; i < GidParamCount; i++) {
				HLgraph.CanonicalStartBlock.Instructions.Insert(GidParamCount + i, new HighLevel.ConditionalBranchInstruction(
					new HighLevel.GreaterEqualsNode(
						new HighLevel.LocationNode(IdLocation[i]),
						new HighLevel.LocationNode(EndIdLocation[i])
					),
					ReturnBlock
					)
				);	
			}

			if (DumpCode > 5) {
				WriteCode(HLgraph, writer);
			}

			// Update location usage information
			HLgraph.AnalyzeLocationUsage();

			// Finally, add the graph to the cache
			CacheEntry = new HlGraphCacheEntry(HLgraph, StartIdLocation, EndIdLocation);

			// Get OpenCL source code
			string OpenClSource;
			using (StringWriter Srcwriter = new StringWriter()) {
				WriteOpenCL(HLgraph, Srcwriter);
				OpenClSource = Srcwriter.ToString();

				if (DumpCode > 2) {
					System.Console.WriteLine(OpenClSource);
				}
			}
			CacheEntry.Source = OpenClSource;

			lock (HlGraphCache) {
				HlGraphCache[Method] = CacheEntry;
			}

			return CacheEntry;
		}

		private static void WriteCode(HighLevel.HlGraph HLgraph, TextWriter writer) {
			writer.WriteLine("// begin {0}", HLgraph.MethodBase);

			if (HLgraph.MethodBase.IsConstructor) {
				writer.Write("constructor {0}::{1} (", ((System.Reflection.ConstructorInfo)HLgraph.MethodBase).DeclaringType, HLgraph.MethodBase.Name);
			} else {
				writer.Write("{0} {1}(", ((MethodInfo)HLgraph.MethodBase).ReturnType, HLgraph.MethodBase.Name);
			}

			for (int i = 0; i < HLgraph.Arguments.Count; i++) {
				if (i > 0) {
					writer.Write(", ");
				}

				HighLevel.ArgumentLocation Argument = HLgraph.Arguments[i];
				string AttributeString = string.Empty;
				if ((Argument.Flags & HighLevel.LocationFlags.IndirectRead) != 0) {
					AttributeString += "__deref_read ";
				}
				if ((Argument.Flags & HighLevel.LocationFlags.IndirectWrite) != 0) {
					AttributeString += "__deref_write ";
				}

				writer.Write("{0}{1} {2}", AttributeString, Argument.DataType, Argument.Name);
			}

			writer.WriteLine(") {");

			foreach (HighLevel.LocalVariableLocation LocalVariable in HLgraph.LocalVariables) {
				writer.WriteLine("\t{0} {1};", LocalVariable.DataType, LocalVariable.Name);
			}

			for (int i = 0; i < HLgraph.BasicBlocks.Count; i++) {
				HighLevel.BasicBlock BB = HLgraph.BasicBlocks[i];

				if (BB == HLgraph.CanonicalEntryBlock || BB == HLgraph.CanonicalExitBlock) {
					continue;
				}

				writer.WriteLine();
				writer.WriteLine("{0}:", BB.LabelName);
				foreach (HighLevel.Instruction Instruction in BB.Instructions) {
					writer.WriteLine("\t{0}", Instruction.ToString());
				}

				if (BB.Successors.Count == 0) {
					writer.WriteLine("\t// unreachable code");
				} else if (i + 1 == HLgraph.BasicBlocks.Count || HLgraph.BasicBlocks[i + 1] != BB.Successors[0]) {
					if (BB.Successors[0] == HLgraph.CanonicalExitBlock) {
						writer.WriteLine("\t// to canonical routine exit");
					} else {
						writer.WriteLine("\tgoto {0};", BB.Successors[0].LabelName);
					}
				}
			}

			writer.WriteLine("}");
			writer.WriteLine("// end");
			writer.WriteLine();
		}

		private static void WriteOpenCL(Type StructType, TextWriter writer) {
			if (StructType == null) {
				throw new ArgumentNullException("StructType");
			} else if (!StructType.IsValueType) {
				throw new ArgumentException(string.Format("Unable to generate OpenCL code for non-ValueType '{0}'", StructType.FullName));
			}

			writer.WriteLine("// OpenCL structure definition for type '{0}'", StructType.FullName);
			writer.WriteLine("struct {0} {{", StructType.Name);
			FieldInfo[] Fields = StructType.GetFields();
			foreach (FieldInfo Field in Fields) {
				writer.WriteLine("\t{0} {1};", GetOpenClType(Field.FieldType), Field.Name);
			}
			writer.WriteLine("}}");
			writer.WriteLine();
		}

		private static string GetOpenCLSourceHeader(OpenCLNet.Platform Platform, OpenCLNet.Device Device) {
			System.Text.StringBuilder String = new System.Text.StringBuilder();

			String.AppendLine("// BEGIN GENERATED OpenCL");

			if (Device.HasExtension("cl_amd_fp64")) {
				String.AppendLine("#pragma OPENCL EXTENSION cl_amd_fp64 : enable");
			} else if (Device.HasExtension("cl_khr_fp64")) {
				String.AppendLine("#pragma OPENCL EXTENSION cl_khr_fp64 : enable");
			}

			if (Device.HasExtension("cl_khr_global_int32_base_atomics")) {
				String.AppendLine("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable");
			}

			String.AppendLine();

			return String.ToString();
		}

		private static void WriteOpenCL(HighLevel.HlGraph HLgraph, TextWriter writer) {
			writer.WriteLine("// OpenCL kernel for method '{0}' of type '{1}'", HLgraph.MethodBase.ToString(), HLgraph.MethodBase.DeclaringType.ToString());
			writer.WriteLine("__kernel {0} {1}(", GetOpenClType(((MethodInfo)HLgraph.MethodBase).ReturnType), HLgraph.MethodName);
			for (int i = 0; i < HLgraph.Arguments.Count; i++) {
				HighLevel.ArgumentLocation Argument = HLgraph.Arguments[i];

				string AttributeString = string.Empty;
				if ((Argument.Flags & HighLevel.LocationFlags.IndirectRead) != 0) {
					AttributeString += "/*[in";
				}
				if ((Argument.Flags & HighLevel.LocationFlags.IndirectWrite) != 0) {
					if (AttributeString == string.Empty) {
						AttributeString += "/*[out";
					} else {
						AttributeString += ",out";
					}
				}

				if (AttributeString != string.Empty) { AttributeString += "]*/ "; }

				if (Argument.DataType.IsArray || Argument.DataType.IsPointer) {
					AttributeString += "__global ";
				}

				writer.WriteLine("\t{0}{1} {2}{3}", AttributeString, GetOpenClType(Argument.DataType), Argument.Name, i + 1 < HLgraph.Arguments.Count ? "," : string.Empty);
			}
			writer.WriteLine(")");
			writer.WriteLine("/*");
			writer.WriteLine("  Generated by CIL2OpenCL");
			writer.WriteLine("*/");
			writer.WriteLine("{");

			foreach (HighLevel.LocalVariableLocation LocalVariable in HLgraph.LocalVariables) {
				string AttributeString = string.Empty;
				if ((LocalVariable.Flags & HighLevel.LocationFlags.Read) != 0) {
					if (AttributeString == string.Empty) { AttributeString += "/*["; } else { AttributeString += ","; }
					AttributeString += "read";
				}
				if ((LocalVariable.Flags & HighLevel.LocationFlags.Write) != 0) {
					if (AttributeString == string.Empty) { AttributeString += "/*["; } else { AttributeString += ","; }
					AttributeString += "write";
				}
				if ((LocalVariable.Flags & HighLevel.LocationFlags.IndirectRead) != 0) {
					if (AttributeString == string.Empty) { AttributeString += "/*["; } else { AttributeString += ","; }
					AttributeString += "deref_read";
				}
				if ((LocalVariable.Flags & HighLevel.LocationFlags.IndirectWrite) != 0) {
					if (AttributeString == string.Empty) { AttributeString += "/*["; } else { AttributeString += ","; }
					AttributeString += "deref_write";
				}
				if (AttributeString == string.Empty) { AttributeString = "/*UNUSED*/ // "; } else { AttributeString += "]*/ "; }

				writer.WriteLine("\t{0}{1} {2};", AttributeString, GetOpenClType(LocalVariable.DataType), LocalVariable.Name);
			}

			HighLevel.BasicBlock FallThroughTargetBlock = HLgraph.CanonicalStartBlock;
			for (int i = 0; i < HLgraph.BasicBlocks.Count; i++) {
				HighLevel.BasicBlock BB = HLgraph.BasicBlocks[i];

				if (BB == HLgraph.CanonicalEntryBlock || BB == HLgraph.CanonicalExitBlock) {
					continue;
				}

				if (FallThroughTargetBlock != null && FallThroughTargetBlock != BB) {
					writer.WriteLine("\tgoto {0};", FallThroughTargetBlock.LabelName);
				}

				FallThroughTargetBlock = null;

				writer.WriteLine();
				if (BB.LabelNameUsed) {
					writer.WriteLine("{0}:", BB.LabelName);
				} else {
					writer.WriteLine("//{0}: (unreferenced block label)", BB.LabelName);
				}

				foreach (HighLevel.Instruction Instruction in BB.Instructions) {
					writer.WriteLine("\t{0}", Instruction.ToString());
				}

				if (BB.Successors.Count == 0) {
					writer.WriteLine("\t// End of block is unreachable");
				} else if (BB.Successors[0] == HLgraph.CanonicalExitBlock) {
					writer.WriteLine("\t// End of block is unreachable/canonical routine exit");
				} else {
					FallThroughTargetBlock = BB.Successors[0];
				}
			}

			writer.WriteLine("}");
			writer.WriteLine("// END GENERATED OpenCL");
		}

		public static string GetOpenClType(Type DataType) {
			return InnerGetOpenClType(DataType);
		}

		private static string InnerGetOpenClType(Type DataType) {
			if (DataType == typeof(void)) {
				return "void";
			} else if (DataType == typeof(sbyte)) {
				return "char";
			} else if (DataType == typeof(byte)) {
				return "uchar";
			} else if (DataType == typeof(short)) {
				return "short";
			} else if (DataType == typeof(ushort)) {
				return "ushort";
			} else if (DataType == typeof(int) || DataType == typeof(IntPtr) || DataType == typeof(bool)) {
				return "int";
			} else if (DataType == typeof(uint) || DataType == typeof(UIntPtr)) {
				return "uint";
			} else if (DataType == typeof(long)) {
				return "long";
			} else if (DataType == typeof(ulong)) {
				return "ulong";
			} else if (DataType == typeof(float)) {
				return "float";
			} else if (DataType == typeof(double)) {
				return "double";
			} else if (DataType.IsArray) {
				return InnerGetOpenClType(DataType.GetElementType()) + "*";
			} else {
				throw new ArgumentException(string.Format("Sorry, data type '{0}' cannot be mapped to OpenCL.", DataType));
			}
		}
	}
}
