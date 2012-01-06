using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.CudaByExample
{
    public class Histogram : ExampleBase // Based on CUDA By Example by Jason Sanders and Edward Kandrot
    {
        byte[] buffer;
        int[] histo;

        protected override void setup()
        {
            this.sizeZ = 256;

            histo = new int[sizeZ];

            buffer = new byte[sizeX];
            for (int i = 0; i < sizeX; i++)
                buffer[i] = (byte)random.Next(0, sizeZ);
        }

        protected override void printInput()
        {
            printField((byte[])buffer, sizeX);
        }

        protected override void algorithm()
        {
            histo = new int[sizeZ];

            int[] temp = new int[sizeZ];

            Parallel.For(ExecuteOn, 0, sizeZ, delegate(int thread_id)
            {
                temp[thread_id] = 0;
            });


            Parallel.For(ExecuteOn, 0, sizeY, delegate(int thread_id)
            {
                int i = thread_id;

                while (i < sizeX)
                {
                    Atomic.Add(ref temp[buffer[i]], 1);
                    i += sizeY;
                }
            });

            Parallel.For(ExecuteOn, 0, sizeZ, delegate(int thread_id)
            {
                Atomic.Add(ref histo[thread_id], temp[thread_id]);
            });
        }

        protected override void printResult()
        {
            printField(histo, sizeZ);
        }

        protected override bool isValid()
        {
            for (int i = 0; i < sizeX; i++)
                histo[buffer[i]]--;

            for (int i = 0; i < sizeZ; i++)
                if (histo[i] != 0)
                    return false;

            return true;
        }
    }
}
