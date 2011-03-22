using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class AccessPathEntry
    {
        public ArgumentLocation ArgumentLocation;
        public Dictionary<System.Reflection.FieldInfo, AccessPathEntry> SubEntries = new Dictionary<System.Reflection.FieldInfo, AccessPathEntry>();
    }
}
