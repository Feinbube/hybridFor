using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Core
{
    // [DataAccess(Access.ReadOnly, Frequency.Once, Pattern.Linear, Stride=3)]
    // [DataAccess(Access.ReadOnly, Frequency.Frequent, Pattern.Arbitrary)]
    // [DataAccess(Access.ReadOnly, Frequency.Once, Pattern.Complex, AffectedPlacesCallback=myCallback)]

    public class DataAccessAttribute : Attribute 
    { 
        public enum Pattern { Linear, Complex, Arbitrary }
        public enum Access { ReadOnly, WriteOnly, ReadWrite }
        public enum Frequency { Never, Once, Seldom, Frequent}

        public delegate List<int> AffectedPlacesDelegate(int id);
        
        public Access access = Access.ReadWrite;
        public Pattern pattern = Pattern.Arbitrary;
        public Frequency frequency = Frequency.Seldom;

        public int Stride = 1;

        public AffectedPlacesDelegate AffectedPlacesCallback;

        public DataAccessAttribute(Access access, Frequency frequency, Pattern pattern)
        {
            this.access = access;
            this.pattern = pattern;
            this.frequency = frequency;
        }
    }
}