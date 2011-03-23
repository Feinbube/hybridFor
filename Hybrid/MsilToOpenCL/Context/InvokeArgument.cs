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
    internal abstract class InvokeArgument : IDisposable
    {
        public abstract object Value { get; }

        public override string ToString()
        {
            object Value = this.Value;
            return (object.ReferenceEquals(Value, null)) ? "<null>" : Value.ToString();
        }

        public abstract void WriteToKernel(CallContext CallContext, int i);
        public abstract void ReadFromKernel(CallContext CallContext, int i);

        #region IDisposable Members

        public virtual void Dispose()
        {
        }

        #endregion

        public class Int32Arg : InvokeArgument
        {
            private int m_Value;

            public Int32Arg(int Value)
            {
                m_Value = Value;
            }

            public override object Value { get { return m_Value; } }
            public override string ToString() { return m_Value.ToString(); }

            public override void WriteToKernel(CallContext CallContext, int i)
            {
                CallContext.Kernel.SetArg(i, m_Value);
            }

            public override void ReadFromKernel(CallContext CallContext, int i)
            {
                // Nothing to do
            }
        }

        public class UInt32Arg : InvokeArgument
        {
            private uint m_Value;

            public UInt32Arg(uint Value)
            {
                m_Value = Value;
            }

            public override object Value { get { return m_Value; } }
            public override string ToString() { return m_Value.ToString(); }

            public override void WriteToKernel(CallContext CallContext, int i)
            {
                CallContext.Kernel.SetArg(i, m_Value);
            }

            public override void ReadFromKernel(CallContext CallContext, int i)
            {
                // Nothing to do
            }
        }

        public class Int64Arg : InvokeArgument
        {
            private long m_Value;

            public Int64Arg(long Value)
            {
                m_Value = Value;
            }

            public override object Value { get { return m_Value; } }
            public override string ToString() { return m_Value.ToString(); }

            public override void WriteToKernel(CallContext CallContext, int i)
            {
                CallContext.Kernel.SetArg(i, m_Value);
            }

            public override void ReadFromKernel(CallContext CallContext, int i)
            {
                // Nothing to do
            }
        }

        public class UInt64Arg : InvokeArgument
        {
            private ulong m_Value;

            public UInt64Arg(ulong Value)
            {
                m_Value = Value;
            }

            public override object Value { get { return m_Value; } }
            public override string ToString() { return m_Value.ToString(); }

            public override void WriteToKernel(CallContext CallContext, int i)
            {
                CallContext.Kernel.SetArg(i, m_Value);
            }

            public override void ReadFromKernel(CallContext CallContext, int i)
            {
                // Nothing to do
            }
        }

        public class FloatArg : InvokeArgument
        {
            private float m_Value;

            public FloatArg(float Value)
            {
                m_Value = Value;
            }

            public override object Value { get { return m_Value; } }
            public override string ToString() { return m_Value.ToString(); }

            public override void WriteToKernel(CallContext CallContext, int i)
            {
                CallContext.Kernel.SetArg(i, m_Value);
            }

            public override void ReadFromKernel(CallContext CallContext, int i)
            {
                // Nothing to do
            }
        }

        public class DoubleArg : InvokeArgument
        {
            private double m_Value;

            public DoubleArg(double Value)
            {
                m_Value = Value;
            }

            public override object Value { get { return m_Value; } }
            public override string ToString() { return m_Value.ToString(); }

            public override void WriteToKernel(CallContext CallContext, int i)
            {
                CallContext.Kernel.SetArg(i, m_Value);
            }

            public override void ReadFromKernel(CallContext CallContext, int i)
            {
                // Nothing to do
            }
        }

        public class ArrayArg : InvokeArgument
        {
            private System.Array m_Value;
            private OpenCLNet.Mem MemBuffer;
            private bool m_ForRead;
            private bool m_ForWrite;
            private System.Runtime.InteropServices.GCHandle? m_GCHandle;

            public ArrayArg(System.Array Value, bool ForRead, bool ForWrite)
            {
                if (!ForRead && !ForWrite)
                {
                    ForRead = ForWrite = true;
                }

                m_Value = Value;
                m_ForRead = ForRead;
                m_ForWrite = ForWrite;
            }

            public override object Value { get { return m_Value; } }
            public override string ToString() { return m_Value.ToString(); }

            public override void WriteToKernel(CallContext CallContext, int i)
            {
                //
                // Get count and size of individual array elements
                //

                long ElementCount = 1;
                for (int d = 0; d < m_Value.Rank; d++)
                {
                    ElementCount *= (m_Value.GetUpperBound(d) - m_Value.GetLowerBound(d) + 1);
                }

                Type RealElementType = Value.GetType().GetElementType();
                int ElementSize = System.Runtime.InteropServices.Marshal.SizeOf(RealElementType);

                //
                // Allocate memory buffer on target hardware using the appropriate type
                //

                OpenCLNet.MemFlags MemFlags;
                if (m_ForRead && !m_ForWrite)
                {
                    MemFlags = OpenCLNet.MemFlags.READ_ONLY;
                }
                else if (!m_ForRead && m_ForWrite)
                {
                    MemFlags = OpenCLNet.MemFlags.WRITE_ONLY;
                }
                else
                {
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

                if (m_ForRead)
                {
                    System.Runtime.InteropServices.GCHandle gch = System.Runtime.InteropServices.GCHandle.Alloc(Value, System.Runtime.InteropServices.GCHandleType.Pinned);

                    try
                    {
                        IntPtr p = gch.Value.AddrOfPinnedObject();
                        CallContext.CommandQueue.EnqueueWriteBuffer(MemBuffer, true, IntPtr.Zero, new IntPtr(ElementCount * ElementSize), p);
                    }
                    finally
                    {
                        gch.Free();
                    }
                }
#endif
            }

            private static void ConvertDoubleToFloatArray(Array DoubleArray, Array FloatArray, int[] dims, int[] widx, int[] ridx, int p)
            {
                if (p == 0)
                {
                    int LowDim = DoubleArray.GetLowerBound(0);
                    for (int i = 0; i < dims[0]; i++)
                    {
                        widx[0] = i;
                        ridx[0] = i + LowDim;
                        FloatArray.SetValue((float)(double)DoubleArray.GetValue(ridx), widx);
                    }
                }
                else
                {
                    int LowDim = DoubleArray.GetLowerBound(p);
                    for (int i = 0; i < dims[p]; i++)
                    {
                        widx[p] = i;
                        ridx[p] = i + LowDim;
                        ConvertDoubleToFloatArray(DoubleArray, FloatArray, dims, widx, ridx, p - 1);
                    }
                }
            }

            public override void ReadFromKernel(CallContext CallContext, int i)
            {
                //
                // Do nothing if the device is guaranteed not to write to this buffer
                //

                if (!m_ForWrite)
                {
                    return;
                }

                //
                // Get count and size of individual array elements
                //

                long ElementCount = 1;
                for (int d = 0; d < m_Value.Rank; d++)
                {
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

            public override void Dispose()
            {
                base.Dispose();
                if (MemBuffer != null)
                {
                    MemBuffer.Dispose();
                    MemBuffer = null;
                }
                if (m_GCHandle.HasValue)
                {
                    m_GCHandle.Value.Free();
                    m_GCHandle = null;
                }
            }

            #endregion
        }
    }
}
