using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL
{
    internal class InvokeContext : IDisposable
    {
        private List<InvokeArgument> m_Arguments;
        public List<InvokeArgument> Arguments
        {
            get
            {
                return m_Arguments;
            }
        }

        public InvokeContext(HighLevel.HlGraph HLgraph)
        {
            m_Arguments = new List<InvokeArgument>(HLgraph.Arguments.Count);
            for (int i = 0; i < HLgraph.Arguments.Count; i++)
            {
                System.Diagnostics.Debug.Assert(HLgraph.Arguments[i].Index == i);
                m_Arguments.Add(null);
            }
        }

        public void PutArgument(HighLevel.ArgumentLocation Location, object Value)
        {
            int Index = Location.Index;

            System.Diagnostics.Debug.Assert(Index >= 0 && Index < m_Arguments.Count);
            System.Diagnostics.Debug.Assert(m_Arguments[Index] == null);

            InvokeArgument Argument = null;
            if (object.ReferenceEquals(Value, null))
            {
                Argument = new InvokeArgument.Int32Arg(0);
            }
            else if (Value is int)
            {
                Argument = new InvokeArgument.Int32Arg((int)Value);
            }
            else if (Value is uint)
            {
                Argument = new InvokeArgument.UInt32Arg((uint)Value);
            }
            else if (Value is long)
            {
                Argument = new InvokeArgument.Int64Arg((long)Value);
            }
            else if (Value is ulong)
            {
                Argument = new InvokeArgument.UInt64Arg((ulong)Value);
            }
            else if (Value is float)
            {
                Argument = new InvokeArgument.FloatArg((float)Value);
            }
            else if (Value is double)
            {
                Argument = new InvokeArgument.DoubleArg((double)Value);
            }
            else
            {
                Type Type = Value.GetType();
                if (Type.IsArray)
                {
                    Type ElementType = Type.GetElementType();
                    if (ElementType == typeof(byte) || ElementType == typeof(int) || ElementType == typeof(uint) || ElementType == typeof(long) || ElementType == typeof(ulong)
                        || ElementType == typeof(float) || ElementType == typeof(double))
                    {
                        bool ForRead = false, ForWrite = false;
                        if ((Location.Flags & HighLevel.LocationFlags.IndirectRead) != 0) { ForRead = true; }
                        if ((Location.Flags & HighLevel.LocationFlags.IndirectWrite) != 0) { ForWrite = true; }

                        if (!ForRead && !ForWrite)
                        {
                            ForRead = ForWrite = true;
                        }

                        Argument = new InvokeArgument.ArrayArg((System.Array)Value, ForRead, ForWrite);
                    }
                }
            }

            if (Argument == null)
            {
                throw new InvalidOperationException(string.Format("Sorry, argument type '{0}' cannot be marshalled for OpenCL.", Value.GetType()));
            }

            m_Arguments[Index] = Argument;
        }

        public void Complete()
        {
            int Index = m_Arguments.IndexOf(null);
            System.Diagnostics.Debug.Assert(Index < 0);
            if (Index >= 0)
            {
                throw new InvalidOperationException(string.Format("Argument {0} is not assigned.", Index));
            }
        }

        #region IDisposable Members

        public void Dispose()
        {
            //foreach (InvokeArgument Argument in m_Arguments)
            //{
            //    if (Argument != null)
            //    {
            //        Argument.Dispose();
            //    }
            //}
            m_Arguments.Clear();
            //System.GC.SuppressFinalize(this);
        }

        #endregion
    }
}
