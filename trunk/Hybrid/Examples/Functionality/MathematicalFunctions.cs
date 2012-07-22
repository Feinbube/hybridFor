using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Functionality
{
    public class MathematicalFunctions: ExampleBase
    {
        double[] da;
        double[] db;
        double[] dc;

        float[] fa;
        float[] fb;
        float[] fc;

        int[] ia;
        int[] ib;
        int[] ic;

        protected override void setup()
        {
            if (sizeX > Int32.MaxValue / 4 || sizeX < 0)
                sizeX = Int32.MaxValue / 4;
            da = new double[sizeX];
            fa = new float[sizeX];
            ia = new int[sizeX];
            for (int i = 0; i < sizeX; i++)
            {
                da[i] = Random.Next() / 7.1f;
                fa[i] = Random.Next() / 7.1f;
                ia[i] = Random.Next();
            }

            db = new double[sizeX];
            fb = new float[sizeX];
            ib = new int[sizeX];
            for (int i = 0; i < sizeX; i++)
            {
                db[i] = Random.Next() / 7.1f;
                fb[i] = Random.Next() / 7.1f;
                ib[i] = Random.Next();

            }

            dc = new double[sizeX];
            fc = new float[sizeX];
            ic = new int[sizeX];

            da[0] = double.MaxValue;
            db[0] = double.MaxValue;

            da[1] = double.MinValue + 5;
            db[1] = double.MinValue + 5;
       
            fa[0] = float.MaxValue;
            fb[0] = float.MaxValue;

            fa[1] = float.MinValue + 5;
            fb[1] = float.MinValue + 5;
            
            ia[0] = int.MaxValue;
            ib[0] = int.MaxValue;

            ia[1] = int.MinValue + 5;
            ib[1] = int.MinValue + 5;
        }

        protected override void printInput()
        {
            //printField(a, sizeX);
            Console.WriteLine();
            //printField(b, sizeX);
        }

        protected override void algorithm()
        {
            Parallel.For(ExecuteOn, 0, sizeX, delegate(int idx)
            {
                //dc[idx] = Math.IEEERemainder(da[idx], db[idx]);
                //fc[idx] = (float)Math.IEEERemainder(fa[idx], fb[idx]);
                //ic[idx] = Math.IEEERemainder(ia[idx], ib[idx]);

                dc[idx] = Math.Abs(da[idx]);
                fc[idx] = (float) Math.Abs(fa[idx]);
                ic[idx] = (int) Math.Abs(ia[idx]);
            });
        }

        protected override void printResult()
        {
            //printField(c, sizeX);
        }

        protected override bool isValid()
        {
            for (int idx = 0; idx < sizeX; idx++)
            {
                if (dc[idx] != Math.Abs(da[idx]) || fc[idx] != (float) Math.Abs(fa[idx]) || ic[idx] != Math.Abs(ia[idx]))
                //if (dc[idx] != Math.IEEERemainder(da[idx], db[idx]) || fc[idx] != (float) Math.IEEERemainder(fa[idx], fb[idx]) /*|| ic[idx] != Math.IEEERemainder(ia[idx], ib[idx])*/)
                    return false;

            }

            return true;
        }

        protected override void cleanup()
        {
            da = null;
            db = null;
            dc = null;
            fa = null;
            fb = null;
            fc = null;
            ia = null;
            ib = null;
            ic = null;
        }
    }
}