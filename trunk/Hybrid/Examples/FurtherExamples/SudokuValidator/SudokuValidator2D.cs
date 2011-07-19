﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class SudokuValidator2D : SudokuValidator
    {
        protected override void algorithm()
        {
            int[] invalidFieldIndicator = new int[4];

            // contains invalid number
            Parallel.For(ExecuteOn, 0, n * n, 0, n * n, delegate(int x, int y)
            {
                int value = field[x + y * n * n];

                if (value > (n * n))
                    invalidFieldIndicator[0] = 1;

                if (value <= 0)
                    invalidFieldIndicator[0] = 1;
            });

            // contains invalid row
            Parallel.For(ExecuteOn, 0, n * n, 0, n * n, delegate(int x, int y)
            {
                for (int x2 = (x + 1); x2 < n * n; x2++)
                    if (field[x + y * n * n] == field[x2 + y * n * n])
                        invalidFieldIndicator[1] = 1; // return true;	
            });

            // contains invalid column
            Parallel.For(ExecuteOn, 0, n * n, 0, n * n, delegate(int y, int x)
            {
                int n2 = n * n;
                
                if (x >= n2 - 1)
                    return;

                for (int y2 = y + 1; y2 < n2; y2++)
                    if (field[x + y * n2] == field[x + y2 * n2])
                        invalidFieldIndicator[2] = 1; // return true;		
            });

            // contains invalid subfield
            Parallel.For(ExecuteOn, 0, n * n, 0, n * n, delegate(int subfield, int cell)
            {
                int n2 = n * n;

                if (cell >= n2 - 1)
                    return;

                for (int cell2 = cell + 1; cell2 < n2; cell2++)
                {
                    int subfield_t1 = (subfield % n) * n;
                    int subfield_t2 = (subfield / n) * n;

                    int x1 = subfield_t1 + (cell % n);
                    int y1 = subfield_t2 + (cell / n);

                    int x2 = subfield_t1 + (cell2 % n);
                    int y2 = subfield_t2 + (cell2 / n);

                    if (field[x1 + y1 * n2] == field[x2 + y2 * n2])
                        invalidFieldIndicator[3] = 1; // return true;				
                }
            });

            isValidField = fieldIsValid(invalidFieldIndicator);
        }
    }
}
