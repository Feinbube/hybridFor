using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class SudokuValidatorInvalidSubfield : SudokuValidator
    {
        protected override void setup()
        {
            n = this.sizeX;

            field = new int[n * n * n * n];
            generateFieldWithInvalidSubfield();
        }

        protected override bool fieldIsValid(int[] invalidFieldIndicator)
        {
            return //invalidFieldIndicator[0] == 0 &&
               //invalidFieldIndicator[1] == 0 &&
               //invalidFieldIndicator[2] == 0 &&
               invalidFieldIndicator[3] == 1;
        }
    }
}
