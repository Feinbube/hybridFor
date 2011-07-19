using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class SudokuValidatorInvalidRow : SudokuValidator
    {
        protected override void setup()
        {
            n = this.sizeX;

            field = new int[n * n * n * n];
            generateFieldWithInvalidRow();
        }

        protected override bool fieldIsValid(int[] invalidFieldIndicator)
        {
            return //invalidFieldIndicator[0] == 0 &&
               invalidFieldIndicator[1] == 1;// &&
               //invalidFieldIndicator[2] == 0 &&
               //invalidFieldIndicator[3] == 0;
        }
    }
}
