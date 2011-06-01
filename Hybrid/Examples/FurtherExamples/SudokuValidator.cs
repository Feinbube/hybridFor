using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class SudokuValidator : ExampleBase
    {
        int[] field;
        int n = 50;
        int shift = 0;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            this.sizeX = this.sizeY = n * n;
            this.sizeZ = -1; // unused
        }

        protected override void setup()
        {
            field = new int[n * n * n * n];
        }

        protected override void printInput()
        {
            throw new NotImplementedException();
        }

        protected override void algorithm()
        {
            bool isValid = false;

            generateSolvedField();

            // warm up
            // for (int i = 0; i < warmUpRounds; i++)
            //     isValid = isValidField(field);

            for (int round = 0; round < sizeZ; round++)
                isValid = isValidField();

        }

        private void generateSolvedField()
        {
            for (int x = 0; x < n * n; x++)
                for (int y = 0; y < n * n; y++)
                    field[coords(x, y, n)] = (x * n + x / n + y + shift) % (n * n) + 1;
        }

        private int coords(int x, int y, int n)
        {
            return x + y * n * n;
        }

        private bool containsInvalidNumberKernel()
        {
            int invalidFieldIndicator = 0;

            Parallel.For(0, n * n, delegate(int id)
            {
                int x = id / (n * n);
                int y = id % (n * n);
                int value = field[x + y * n * n];

                if (value > (n * n) || value <= 0)
                    invalidFieldIndicator = 1;
            });

            return invalidFieldIndicator != 0;
        }

        private bool containsInvalidRowKernel()
        {
            int invalidFieldIndicator = 0;

            Parallel.For(0, n * n, delegate(int id)
            {
                int x = id / (n * n);
                int y = id % (n * n);

                for (int x2 = (x + 1); x2 < n * n; x2++)
                    if (field[x + y * n * n] == field[x2 + y * n * n])
                        invalidFieldIndicator = 2; // return true;	
            });

            return invalidFieldIndicator != 0;
        }

        private bool containsInvalidSubfieldKernel()
        {
            int invalidFieldIndicator = 0;

            Parallel.For(0, n * n, delegate(int threadId)
            {
                int n2 = n * n;

                // for (int subfield = 0; subfield < n*n; subfield++)           
                int subfield = threadId / (n2);

                // for (int cell = 0; cell < n*n-1; cell++)                     
                int cell = threadId % (n2);
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
                        invalidFieldIndicator = 4; // return true;				
                }
            });

            return invalidFieldIndicator != 0;
        }

        private bool containsInvalidColumnKernel()
        {
            int invalidFieldIndicator = 0;

            Parallel.For(0, n * n, delegate(int threadId)
            {
                int n2 = n * n;

                // for (int y = 0; y < n2; y++)                             
                int y = threadId / (n2);

                // for (int x = 0; x < n2-1; x++)                            
                int x = threadId % (n2);
                if (x >= n2 - 1)
                    return;

                for (int y2 = y + 1; y2 < n2; y2++)
                    if (field[x + y * n2] == field[x + y2 * n2])
                        invalidFieldIndicator = 3; // return true;		
            });

            return invalidFieldIndicator != 0;
        }

        private bool isValidField()
        {
            bool invalid = containsInvalidNumberKernel()
                || containsInvalidRowKernel()
                || containsInvalidColumnKernel()
                || containsInvalidSubfieldKernel();

            return !invalid;
        }

        protected override void printResult()
        {
            throw new NotImplementedException();
        }

        protected override bool isValid()
        {
            return false;
        }
    }
}
