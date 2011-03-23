using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.MsilToOpenCL.HighLevel
{
    public class LocationNode : Node
    {
        private Location m_Location;

        public LocationNode(Location Location)
            : base(NodeType.Location, Location.DataType, true)
        {
            m_Location = Location;
        }

        public Location Location
        {
            get
            {
                return m_Location;
            }
            set
            {
                m_Location = value;
            }
        }

        public override Type DataType
        {
            get
            {
                return base.DataType;
            }
            set
            {
                base.DataType = value;
                if (m_Location != null && m_Location.DataType == null)
                {
                    m_Location.DataType = value;
                }
            }
        }

        public override string ToString()
        {
            return m_Location.Name;
        }
    }
}
