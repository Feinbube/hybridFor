using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace Hybrid.MsilToOpenCL
{
    /// <summary>
    /// This attribute marks methods that represent built-in OpenCL functions.
    /// If the Alias property is not specified, the .NET name of the method is used as-is in
    /// OpenCL. Otherwise, if the Alias is present, its value is used instead of the internal
    /// .NET name.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
    public sealed class OpenClAliasAttribute : Attribute
    {
        private string m_Alias;
        public OpenClAliasAttribute()
            : this(string.Empty)
        {
        }

        public OpenClAliasAttribute(string Alias)
        {
            m_Alias = Alias;
        }

        public string Alias { get { return m_Alias; } }

        public static string Get(MethodInfo MethodInfo)
        {
            if (MethodInfo == null)
            {
                throw new ArgumentNullException("MethodInfo");
            }

            object[] Alias = MethodInfo.GetCustomAttributes(typeof(OpenClAliasAttribute), false);
            if (Alias.Length == 0)
            {
                return null;
            }
            else if (string.IsNullOrEmpty(((OpenClAliasAttribute)Alias[0]).Alias))
            {
                return MethodInfo.Name;
            }
            else
            {
                return ((OpenClAliasAttribute)Alias[0]).Alias;
            }
        }
    }
}
