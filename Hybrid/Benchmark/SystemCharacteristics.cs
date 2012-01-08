﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Hybrid.Examples;
using System.Xml.Serialization;
using System.IO;

namespace Hybrid.Benchmark
{
    public class SystemCharacteristics
    {
        public class DoublePair
        {
            public double Key;
            public double Value;
        }

        public class Example
        {
            public string Name;
            public List<DoublePair> AppropriateScales;

            public DoublePair Get(ExampleBase exampleBase, double minSequentialExecutionTime)
            {
                foreach (DoublePair doublePair in AppropriateScales)
                    if (doublePair.Key == minSequentialExecutionTime)
                        return doublePair;

                DoublePair result = new DoublePair();
                result.Key = minSequentialExecutionTime;
                result.Value = findGoodSize(exampleBase, minSequentialExecutionTime);

                AppropriateScales.Add(result);

                return result;
            }

            private double findGoodSize(ExampleBase example, double minSequentialExecutionTime)
            {
                example.ExecuteOn = Execute.OnSingleCpu;

                double scale = 5;

                double executionTime = example.Run(scale, scale, scale, false, 20, 5).ElapsedTotalSeconds;
                while (executionTime <= 0.001)
                {
                    scale *= 2;
                    executionTime = example.Run(scale, scale, scale, false, 20, 5).ElapsedTotalSeconds;
                }

                double executionTime2 = example.Run(scale * 2, scale * 2, scale * 2, false, 20, 5).ElapsedTotalSeconds;

                double log = Math.Log(executionTime2/executionTime, 2);
                scale = Math.Pow(minSequentialExecutionTime / executionTime, 1/log) * scale;

                // executionTime = example.Run(scale, scale, scale, false, 20, 5).ElapsedTotalSeconds;

                Console.WriteLine("Scale " + scale + " for " + example.GetType().Name + "."); // + " for " + executionTime + "s.");

                return scale;
            }
        }

        public List<Example> Examples;
        
        public string MachineName { get; set; }
        public Platform Platform { get; set; }

        public SystemCharacteristics()
        {
            Examples = new List<Example>();
            
            MachineName = Environment.MachineName;
            Platform = Hybrid.Scheduler.Platform;
        }

        private Example Get(string exampleName)
        {
            foreach (Example example in Examples)
                if (example.Name == exampleName)
                    return example;

            Example result = new Example();
            result.Name = exampleName;
            result.AppropriateScales = new List<DoublePair>();

            Examples.Add(result);

            return result;
        }

        public double GetScale(ExampleBase exampleBase, double minSequentialExecutionTime)
        {
            Example example = Get(exampleBase.GetType().ToString());
            DoublePair pair = example.Get(exampleBase, minSequentialExecutionTime);

            return pair.Value;
        }

        public override string ToString()
        {
            string result = "";

            result = "Compute Devices found on " + MachineName + ":\r\n";

            foreach (ComputeDevice computeDevice in Platform.ComputeDevices)
                result += " * " + computeDevice.Name + " [performance index: " + computeDevice.PredictPerformance(new AlgorithmCharacteristics()) + "]" + "\r\n";

            return result;
        }
    }
}
