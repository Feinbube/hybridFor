/*    
*    AtomicExample.cs
*
﻿*    Copyright (C) 2012  Frank Feinbube, Jan-Arne Sobania, Ralf Diestelkämper
*
*    This library is free software: you can redistribute it and/or modify
*    it under the terms of the GNU Lesser General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public License
*    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*    Frank [at] Feinbube [dot] de
*    jan-arne [dot] sobania [at] gmx [dot] net
*    ralf [dot] diestelkaemper [at] hotmail [dot] com
*
*/


﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Functionality
{
    public class AtomicExample : ExampleBase
    {
        int[] a;
        int result;
        protected override void setup()
        {
            if (sizeX > Int32.MaxValue || sizeX < 0)
                sizeX = Int32.MaxValue;
            a = new int[sizeX];
            for (int x = 0; x < sizeX; x++)
            {
                a[x] = 1;
            }
            result = 0;
        }

        protected override void printInput()
        {
            printField(a, sizeX);
        }

        protected override void algorithm()
        {
            result = 0;
            Parallel.For(ExecuteOn,0,sizeX,delegate(int x){
                Atomic.Add(ref result, a[x]);
            });
        }

        protected override void printResult()
        {
            Console.WriteLine(result);
        }


        protected override void cleanup()
        {
            a = null;
        }

        protected override bool isValid()
        {
            if (result == sizeX)
            {
                return true;
            }
            return false;
        }

    }
}
