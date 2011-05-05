﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples.Upcrc2010.MatrixMultiplication
{
    public class MatrixMultiplication5 : MatrixMultiplicationBase
    {
        protected override void scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            double factor = 125.0;
            this.sizeX = (int)(sizeX * factor);
            this.sizeY = (int)(sizeY * factor);
            this.sizeZ = (int)(sizeZ * factor);
        }

        protected override void algorithm() // http://myssa.upcrc.illinois.edu/files/Lab_OpenMP_Assignments/
        {
            int i, j;

            for (i = 0; i < sizeX; i++)
                for (j = 0; j < sizeY; j++)
                    c[i, j] = 0;

            matmultrec(sizeX, sizeY, sizeZ, 0, sizeX, 0, sizeY, 0, sizeZ, a, b, c);
        }

        void matmultrec(int m, int n, int p, int mf, int ml, int nf, int nl, int pf, int pl, double[,] A, double[,] B, double[,] C)
        {
            if ((ml - mf) * (nl - nf) * (pl - pf) < 8 * 32768) /* product size below which matmultleaf is used */
                matmultleaf(m, n, p, mf, ml, nf, nl, pf, pl, A, B, C);
            else
            {
                matmultrec(m, n, p, mf, mf + (ml - mf) / 2, nf, nf + (nl - nf) / 2, pf, pf + (pl - pf) / 2, A, B, C);
                matmultrec(m, n, p, mf, mf + (ml - mf) / 2, nf + (nl - nf) / 2, nl, pf, pf + (pl - pf) / 2, A, B, C);
                matmultrec(m, n, p, mf, mf + (ml - mf) / 2, nf, nf + (nl - nf) / 2, pf + (pl - pf) / 2, pl, A, B, C);
                matmultrec(m, n, p, mf, mf + (ml - mf) / 2, nf + (nl - nf) / 2, nl, pf + (pl - pf) / 2, pl, A, B, C);
                matmultrec(m, n, p, mf + (ml - mf) / 2, ml, nf, nf + (nl - nf) / 2, pf, pf + (pl - pf) / 2, A, B, C);
                matmultrec(m, n, p, mf + (ml - mf) / 2, ml, nf + (nl - nf) / 2, nl, pf, pf + (pl - pf) / 2, A, B, C);
                matmultrec(m, n, p, mf + (ml - mf) / 2, ml, nf, nf + (nl - nf) / 2, pf + (pl - pf) / 2, pl, A, B, C);
                matmultrec(m, n, p, mf + (ml - mf) / 2, ml, nf + (nl - nf) / 2, nl, pf + (pl - pf) / 2, pl, A, B, C);
            }
        }

        void matmultleaf(int m, int n, int p, int mf, int ml, int nf, int nl, int pf, int pl, double[,] A, double[,] B, double[,] C)
        {
            Parallel.For(mf, ml, nf, nl, delegate(int i, int j)
            {
                for (int k = pf; k < pl; k++)
                    C[i, j] += A[i, k] * B[k, j];
            });
        }
    }
}
