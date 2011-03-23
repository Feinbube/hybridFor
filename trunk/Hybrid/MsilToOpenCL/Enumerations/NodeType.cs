using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public enum NodeType
    {
        StringConstant,
        IntegerConstant,
        FloatConstant,
        DoubleConstant,
        Location,

        InstanceField,

        Call,
        ArrayAccess,
        Cast,

        Neg,
        LogicalNot,
        AddressOf,
        Deref,

        Equals,
        NotEquals,
        Less,
        LessEquals,
        Greater,
        GreaterEquals,

        Add,
        Sub,
        Mul,
        Div,
        Mod,
    }
}
