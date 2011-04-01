using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Management;
using System.Runtime.InteropServices;

namespace Hybrid
{
    public class ComputeDevice
    {
        public enum DeviceTypes { NumaNode, Gpu, Accelerator, Unknown }
        public DeviceTypes DeviceType;

        public string Name;
        public string Manufacturer;
        public string DeviceId;

        public MemoryInfo GlobalMemory;

        public List<ComputeUnit> ComputeUnits;

        [DllImport("kernel32.dll", EntryPoint = "GetNumaAvailableMemoryNodeEx")]
        [return: MarshalAsAttribute(UnmanagedType.Bool)]
        public static extern bool GetNumaAvailableMemoryNodeEx(byte Node, out ulong AvailableBytes);

        public ComputeDevice(ManagementObject processorInfo)
        {

            ulong size;

            for(byte i=0; i<10; i++)
                GetNumaAvailableMemoryNodeEx(i, out size);



            DeviceType = DeviceTypes.NumaNode;

            Name = processorInfo["Name"].ToString();
            Manufacturer = processorInfo["Manufacturer"].ToString();
            DeviceId = processorInfo["DeviceID"].ToString();

            uint NumberOfCores = uint.Parse(processorInfo["NumberOfCores"].ToString());

            ComputeUnits = new List<ComputeUnit>();
            for (int i = 0; i < NumberOfCores; i++)
                ComputeUnits.Add(new ComputeUnit(processorInfo));



            //GlobalMemory = new MemoryInfo()
            //{
            //    Size = device.GlobalMemSize,

            //    CacheType = mapCacheType(device.GlobalMemCacheType),
            //    CacheSize = device.GlobalMemCacheSize,
            //    CacheLineSize = device.GlobalMemCacheLineSize
            //};
        }

        public ComputeDevice(OpenCLNet.Device device)
        {
            DeviceType = getDeviceType(device);

            Name = device.Name;
            Manufacturer = device.Vendor;

            DeviceId = device.DeviceID.ToString(); // TODO: get real device id from handle

            ComputeUnits = new List<ComputeUnit>();
            for (int i = 0; i < device.MaxComputeUnits; i++)
                ComputeUnits.Add(new ComputeUnit(device));

            GlobalMemory = new MemoryInfo()
            {
                Size = device.GlobalMemSize,

                CacheType = mapCacheType(device.GlobalMemCacheType),
                CacheSize = device.GlobalMemCacheSize,
                CacheLineSize = device.GlobalMemCacheLineSize
            };
        }

        private ComputeDevice.DeviceTypes getDeviceType(OpenCLNet.Device device)
        {
            ComputeDevice.DeviceTypes deviceType = ComputeDevice.DeviceTypes.Unknown;

            if (device.DeviceType == OpenCLNet.DeviceType.CPU)
                deviceType = ComputeDevice.DeviceTypes.NumaNode;

            if (device.DeviceType == OpenCLNet.DeviceType.GPU)
                deviceType = ComputeDevice.DeviceTypes.Gpu;

            if (device.DeviceType == OpenCLNet.DeviceType.ACCELERATOR)
                deviceType = ComputeDevice.DeviceTypes.Accelerator;

            return deviceType;
        }

        private MemoryInfo.CacheTypes mapCacheType(OpenCLNet.DeviceMemCacheType deviceMemCacheType)
        {
            if (deviceMemCacheType == OpenCLNet.DeviceMemCacheType.READ_ONLY_CACHE)
                return MemoryInfo.CacheTypes.ReadOnly;

            if (deviceMemCacheType == OpenCLNet.DeviceMemCacheType.READ_WRITE_CACHE)
                return MemoryInfo.CacheTypes.ReadWrite;

            if (deviceMemCacheType == OpenCLNet.DeviceMemCacheType.NONE)
                return MemoryInfo.CacheTypes.None;

            throw new Exception("OpenCLNet.DeviceMemCacheType " + deviceMemCacheType + " unknown.");
        }
    }
}
