﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class GameOfLife: ExampleBase
    {
        byte[,] state1;
        byte[,] state2;

        protected override void setup()
        {
            this.sizeZ = 10;

            state1 = new byte[sizeX, sizeY];
            state2 = new byte[sizeX, sizeY];

            for (int x = 1; x < sizeX-1; x++)
                for (int y = 1; y < sizeY-1; y++)
                    state1[x, y] = (byte)random.Next(2);
        }

        private void printField(byte[,] fields)
        {
            printField(fields, sizeX, sizeY, delegate(int x, int y)
            {
                if (state1[x, y] == 0)
                    Console.Write(" ");
                else
                    Console.Write("X");
            });
        }

        protected override void printInput()
        {
            printField(state1);
        }

        protected override void algorithm()
        {
            for (int z=0; z<sizeZ; z++) {
                Parallel.For(ExecuteOn, 1, sizeX - 1, 1, sizeY - 1, delegate(int x, int y)
                {
                    int livingNeighbors = 0;

                    livingNeighbors += state1[x-1, y-1];
                    livingNeighbors += state1[x, y-1];
                    livingNeighbors += state1[x+1, y-1];
                    livingNeighbors += state1[x-1, y];
                    livingNeighbors += state1[x+1, y];
                    livingNeighbors += state1[x-1, y+1];
                    livingNeighbors += state1[x, y+1];
                    livingNeighbors += state1[x+1, y+1];

                    if ((state1[x, y] == 1 && livingNeighbors == 2) || livingNeighbors == 3)
                        state2[x, y] = 1;
                    else
                        state2[x, y] = 0;
                });
            
                swap(ref state1, ref state2);
            }
        }

        protected override void printResult()
        {
            printField(state2);
        }

        protected override bool isValid()
        {
            return true;
        }

        protected override void cleanup()
        {
            state1 = null;
            state2 = null;
        }
    }
}
