using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Core
{
    // [DataAccess(Access.ReadOnly, Pattern.Linear, Stride=3)]
    // [DataAccess(Access.ReadOnly, Pattern.Arbitrary)]
    // [DataAccess(Access.ReadOnly, Pattern.Complex, AffectedPlacesCallback=myCallback)]

    public class DataAccessAttribute : Attribute 
    { 
        public enum Pattern { Linear, Complex, Arbitrary }
        public enum Access { ReadOnly, WriteOnly, ReadWrite }
        public delegate List<int> AffectedPlacesDelegate(int id);
        
        public Access access = Access.ReadWrite;
        public Pattern pattern = Pattern.Arbitrary;

        public int Stride = 1;
        public AffectedPlacesDelegate AffectedPlacesCallback;

        public DataAccessAttribute(Access access, Pattern pattern)
        {
            this.access = access;
            this.pattern = pattern;
        }
    }
}
