using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class SudokuValidator: ExampleBase
    {
        int[] field;
        int n, shift;

        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            this.sizeX = -1; // unused
            this.sizeY = -1; // unused
            this.sizeZ = 1; // Used to store the number of rounds

            this.n = 20;
            this.shift = 0;
        }

        protected override void setup()
        {
            field = new int[sizeX * sizeY];


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

        private bool containsInvalidRow()
        {
            int invalidFieldIndicator = 0;

            Parallel.For(0, n * n, delegate(int id)
            {
                int x = id / (n * n);
                int y = id % (n * n);

               for (int x2 = (x+1); x2 < n*n; x2++)						
		            if(field[(x)+((y)*n*n)] == field[x2+y*n*n])	 		
			            invalidFieldIndicator = 2; // return true;	
            });

            return invalidFieldIndicator != 0;
        }

        private bool isValidField()
        {
            bool containsInvalidNumberKernel = false;
            bool containsInvalidRowKernel = false;
            bool containsInvalidColumnKernel = false;
            bool containsInvalidSubfieldKernel = false;

            // TODO calculate each boolean values in parallel because the field is big.

            bool invalid = containsInvalidNumberKernel
                || containsInvalidRowKernel
                || containsInvalidColumnKernel
                || containsInvalidSubfieldKernel;

            return !invalid;
        }

        protected override void printResult()
        {
            throw new NotImplementedException();
        }

        protected override bool isValid()
        {
            throw new NotImplementedException();
        }
    }
}
