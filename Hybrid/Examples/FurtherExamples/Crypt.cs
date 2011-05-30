using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Hybrid.Examples
{
    public class Crypt: ExampleBase
    {
        int[][] S = new int[8][]
        {
            new int[]{
                    14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7,
                     0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8,
                     4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0,
                    15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13
                }
            ,
                new int[]{
                15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10,
                 3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5,
                 0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15,
                13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9
                }
            ,
                new int[]{
                10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8,
                13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1,
                13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7,
                 1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12
            },
            new int[]{
                 7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15,
                13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9,
                10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4,
                 3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14
            },
            new int[]{
                 2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9,
                14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6,
                 4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14,
                11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3
            },
            new int[]{
                12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11,
                10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8,
                 9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6,
                 4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13
            },
            new int[]{
                 4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1,
                13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6,
                 1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2,
                 6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12
            },
            new int[]{
                13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7,
                 1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2,
                 7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8,
                 2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11
            }    
        };

        /* Initial permutation */
         int[] IP = new int[]
        {
            58, 50, 42, 34, 26, 18, 10, 2,
            60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6,
            64, 56, 48, 40, 32, 24, 16, 8,
            57, 49, 41, 33, 25, 17,  9, 1,
            59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5,
            63, 55, 47, 39, 31, 23, 15, 7,
        };

        /* Final permutation, FP = IP^(-1) */
         int[] FP = new int[]{
            40, 8, 48, 16, 56, 24, 64, 32,
            39, 7, 47, 15, 55, 23, 63, 31,
            38, 6, 46, 14, 54, 22, 62, 30,
            37, 5, 45, 13, 53, 21, 61, 29,
            36, 4, 44, 12, 52, 20, 60, 28,
            35, 3, 43, 11, 51, 19, 59, 27,
            34, 2, 42, 10, 50, 18, 58, 26,
            33, 1, 41,  9, 49, 17, 57, 25,
        };

        /**************************************************************************
        * Permuted-choice 1 from the key bits to yield C and D.
        * Note that bits 8,16... are left out:
        * They are intended for a parity check.
        **************************************************************************/
         int[] PC1_C = new int[]
        {
            57, 49, 41, 33, 25, 17,  9,
             1, 58, 50, 42, 34, 26, 18,
            10,  2, 59, 51, 43, 35, 27,
            19, 11,  3, 60, 52, 44, 36,
        };

         int[] PC1_D = new int[]
        {
            63, 55, 47, 39, 31, 23, 15,
             7, 62, 54, 46, 38, 30, 22,
            14,  6, 61, 53, 45, 37, 29,
            21, 13,  5, 28, 20, 12,  4,
        };

        /* Sequence of shifts used for the key schedule. */
         int[] shifts = 
            {1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1};

        /**************************************************************************
        * Permuted-choice 2, to pick out the bits from the CD array that generate
        * the key schedule.
        **************************************************************************/
         int[] PC2_C = new int[]{
            14, 17, 11, 24,  1,  5,
             3, 28, 15,  6, 21, 10,
            23, 19, 12,  4, 26,  8,
            16,  7, 27, 20, 13,  2,
        };

         int[] PC2_D = new int[]{
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32,
        };

        /* The E bit-selection table. */
         int[] e2 = new int[]{
            32,  1,  2,  3,  4,  5,
             4,  5,  6,  7,  8,  9,
             8,  9, 10, 11, 12, 13,
            12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21,
            20, 21, 22, 23, 24, 25,
            24, 25, 26, 27, 28, 29,
            28, 29, 30, 31, 32,  1,
        };

        /**************************************************************************
        * P is a permutation on the selected combination of the current L and key.
        **************************************************************************/
         int[] P = new int[]
        {
            16,  7, 20, 21,
            29, 12, 28, 17,
             1, 15, 23, 26,
             5, 18, 31, 10,
             2,  8, 24, 14,
            32, 27,  3,  9,
            19, 13, 30,  6,
            22, 11,  4, 25,
        };

        struct PwdEntry
        {
            public char[] username;
            public char[] salt;
            public char[] password;
            public int cracked;

            public PwdEntry(char[] username, char[] salt, char[] password)
            {
                this.username = username;
                this.salt = salt;
                this.password = password;
                this.cracked = 0;
            }
        }

        char[][] dict;
        PwdEntry[] pwd;
        bool[] pwCracked;

        protected override void  scaleAndSetSizes(double sizeX, double sizeY, double sizeZ)
        {
            this.sizeX = this.sizeY = this.sizeZ = -1;
        }

        protected override void  setup()
        {
            /*
             * results:
             * user266 - Osten3
             * user906 - Bahnhof
             */

            pwd = new PwdEntry[]{
                new PwdEntry("user266".ToCharArray(), "EB".ToCharArray(), "EBCi4DBY9TjUk".ToCharArray()),
                new PwdEntry("user906".ToCharArray(), "iu".ToCharArray(), "iuAROc/9zKe0I".ToCharArray())
            };

            dict = new char[][] {
                new char[] { 'O', 's', 't', 'e', 'n', '3', (char) 0, },
                new char[] { 'B', 'a', 'h', 'n', 'h', 'o','f', (char) 0 }
            };
            pwCracked = new bool[2];
        }

        protected override void  printInput()
        {
            throw new NotImplementedException();
        }

        protected override void  algorithm()
        {
            int pwdEntryCount = pwd.Length;

            Parallel.For(0, dict.Length, delegate(int dictId)
            {
                char[] password = dict[dictId];
                char[] newPassword = new char[16];
                char[] numberedPassword = new char[8];

                int length = memcpy_pg(password, dict[dictId], 8);
                bool tryExtendedPasswords = length < 8;

                // try this dictionary entry with each user
                for(int i = 0; i < pwdEntryCount; i++)
                {
                   PwdEntry curEntry = pwd[i]; // __private PWDENTRY curEntry = pwd[i];
                   bool result = pwCracked[i]; // __global char* result = crkPwd + (i*9);

                   if (result == false) { // if(result[0] == 0)
                    {
                        // encrypt
                        crypt(password, curEntry.salt, newPassword);

                        if(strcmp(newPassword, curEntry.password) == 0)
                        {
                            // mark user as cracked and copy password to result store
                            pwCracked[i] = true;
                            Console.WriteLine("pw = " + newPassword);
//                            memcpy_gp(result + 1, password, 8);
                            break;
                        }

                        // try extended passwords
                        if(tryExtendedPasswords)
                        {
                            memcpy_pp(numberedPassword, password, 8);

                            for(char k = '0'; k <= '9'; k++)
                            {
                                numberedPassword[length - 1] = k;						
                        
                                // encrypt
                                crypt(numberedPassword, curEntry.salt, newPassword);

                                if(strcmp(newPassword, curEntry.password) == 0)
                                {
                                    // mark user as cracked and copy password to result store
                                    pwCracked[i] = true;
                                    Console.WriteLine("pw = " + newPassword);
//                                    memcpy_gp(result + 1, numberedPassword, 8);
                                    break;
                                }
                            }
                        }
                    }
                   }
                }	
            });
    }

    private int strcmp(char[] str1, char[] str2)
    {
        int i = 0;
        while(str1[i] != 0 && str2[i] != 0)
        {
            if(str1[i] != str2[i])
                return 1;

            i++;
        }

        return 0;
    }

    private int memcpy_pg(char[] dst, char[] src, int len)
    {
        int length = 0;
        for(int i = 0; i<len; i++)
        {
            dst[i] = src[i];
            if(dst[i] == 0 && length == 0)
                length = i;
        }

        if(length == 0)
            return len;
        else
            return length + 1;
    }

    private void memcpy_gp(char[] dst, char[] src, int len)
    {
        for(int i = 0; i<len; i++)
            dst[i] = src[i];
    }
        
    private int memcpy_pp(char[] dst, char[] src, int len)
    {
        int length = 0;
        for(int i = 0; i<len; i++)
        {
            dst[i] = src[i];
            if(dst[i] == 0 && length == 0)
                length = i;
        }

        if(length == 0)
            return len;
        else
            return length + 1;
    }

        private void crypt(char[] pw, char[] salt, char[] iobuf)
        {
            /* The C and D arrays used to calculate the key schedule. */
            char[] C = new char[28];
            char[] D = new char[28];

            /* The key schedule.  Generated from the key. */
            char[][] KS = new char[16][/* 48 */];
            for (int ks_i = 0; ks_i < 16; ks_i++)
                KS[ks_i] = new char[48];

            /* The E bit-selection table. */
            char[] E = new char[48];

            /* The combination of the key and the input, before selection. */
            char[] preS = new char[48];

            char i, j, temp;
            char c;
            char[] block = new char[66];

            for(i = (char)0; i < 66; i++)
                block[i] = (char) 0;

            /* break pw into 64 bits */
            for(i = (char)0, c = pw[0]; c != 0 && (i < 64); i++)
            {
                for(j = (char)0; j < 7; j++, i++)
                    block[i] = (char) ((c >> (6 - j)) & 01); // TODO

                if (i<pw.Length-1)
                    c = pw[i + 1];
            }

            /* set key based on pw */
            setkey(block, C, D, E, KS);

            for(i = (char) 0; i < 66; i++)
            {
                block[i] = (char) 0;
            }

            for(i = (char) 0; i < 2; i++)
            {
                /* store salt at beginning of results */
                c = salt[i];
                //iobuf[i] = c;

                if(c > 'Z')
                    c -= (char) 6;

                if(c > '9')
                    c -= (char) 7;

                c -= '.';

                /* use salt to effect the E-bit selection */
                for(j = (char) 0; j < 6; j++)
                {
                    if(((c >> j) & 01) != 0)
                    {
                        temp = E[6 * i + j];
                        E[6 * i +j] = E[6 * i + j + 24];
                        E[6 * i + j + 24] = temp;
                    }
                }
            }

            /* call DES encryption 25 times using pw as key and initial data = 0 */
            for(i = (char) 0; i < 25; i++)
                encrypt(block, E, KS, preS);

            /* format encrypted block for standard crypt(3) output */
            for(i= (char) 0; i < 11; i++)
            {
                c = (char) 0;
                for(j = (char)0; j < 6; j++)
                {
                    c <<= 1;
                    c |= block[6 * i + j];
                }

                c += '.';
                if(c > '9')
                    c += (char) 7;

                if(c > 'Z')
                    c += (char) 6;

                iobuf[i /* + 2 */] = c;
            }

            iobuf[i /* + 2 */] = '\0';

            /* prevent premature NULL terminator */
            if(iobuf[1] == '\0')
                iobuf[1] = iobuf[0];
        }

        private void encrypt(char[] block,char[] E,char[][] KS, char[] preS)
        {
            
            int i, ii, temp, j, k;

            char[] left = new char[32];
            char[] right = new char[32]; /* block in two halves */
            char[] old = new char[32];
            char[] f = new char[32];

            /* First, permute the bits in the input */
            for(j = 0; j < 32; j++)
                left[j] = block[IP[j] - 1];

            for(;j < 64; j++)
                right[j - 32] = block[IP[j] - 1];

            /* Perform an encryption operation 16 times. */
            for(ii= 0; ii < 16; ii++)
            {
                i = ii;
                /* Save the right array, which will be the new left. */
                for(j = 0; j < 32; j++)
                    old[j] = right[j];

                /******************************************************************
                * Expand right to 48 bits using the E selector and
                * exclusive-or with the current key bits.
                ******************************************************************/
                for(j =0 ; j < 48; j++)
                    preS[j] = (char) (right[E[j] - 1] ^ KS[i][j]); // TODO

                /******************************************************************
                * The pre-select bits are now considered in 8 groups of 6 bits ea.
                * The 8 selection functions map these 6-bit quantities into 4-bit
                * quantities and the results are permuted to make an f(R, K).
                * The indexing into the selection functions is peculiar;
                * it could be simplified by rewriting the tables.
                ******************************************************************/
                for(j = 0; j < 8; j++)
                {
                    temp = 6 * j;
                    k = S[j][(preS[temp + 0] << 5) +
                        (preS[temp + 1] << 3) +
                        (preS[temp + 2] << 2) +
                        (preS[temp + 3] << 1) +
                        (preS[temp + 4] << 0) +
                        (preS[temp + 5] << 4)];

                    temp = 4 * j;

                    f[temp + 0] = (char) ((k >> 3) & 01); // TODO
                    f[temp + 1] = (char) ((k >> 2) & 01); // TODO
                    f[temp + 2] = (char) ((k >> 1) & 01); // TODO
                    f[temp + 3] = (char) ((k >> 0) & 01); // TODO
                }

                /******************************************************************
                * The new right is left ^ f(R, K).
                * The f here has to be permuted first, though.
                ******************************************************************/
                for(j = 0; j < 32; j++)
                    right[j] = (char) (left[j] ^ f[P[j] - 1]);

                /* Finally, the new left (the original right) is copied back. */
                for(j = 0; j < 32; j++)
                    left[j] = old[j];
            }

            /* The output left and right are reversed. */
            for(j = 0; j < 32; j++)
            {
                temp = left[j];
                left[j] = right[j];
                right[j] = (char) temp;
            }

            /* The final output gets the inverse permutation of the very original. */
            for(j = 0; j < 64; j++)
            {
                i = FP[j];
                if (i < 33)
                        block[j] = left[FP[j] - 1];
                else
                        block[j] = right[FP[j] - 33];
            }
        }

        private void setkey(char[] key,char[] C,char[] D,char[] E,char[][] KS)
        {
            int i, j, k;
            char temp;

            /**********************************************************************
            * First, generate C and D by permuting the key.  The low order bit of
            * each 8-bit char is not used, so C and D are only 28 bits apiece.
            **********************************************************************/
            for(i = 0; i < 28; i++)
            {
                C[i] = key[PC1_C[i] - 1];
                D[i] = key[PC1_D[i] - 1];
            }

            /**********************************************************************
            * To generate Ki, rotate C and D according to schedule and pick up a
            * permutation using PC2.
            **********************************************************************/
            for(i = 0; i < 16; i++)
            {
                /* rotate */
                for(k = 0; k < shifts[i]; k++)
                {
                    temp = C[0];

                    for(j = 0; j < 28 - 1; j++)
                        C[j] = C[j+1];

                    C[27] = temp;
                    temp = D[0];
                    for(j = 0; j < 28 - 1; j++)
                        D[j] = D[j+1];

                    D[27] = temp;
                }

                /* get Ki. Note C and D are concatenated */
                for(j = 0; j < 24; j++)
                {
                    KS[i][j] = C[PC2_C[j] - 1];
                    KS[i][j + 24] = D[PC2_D[j] - 28 -1];
                }
            }

            /* load E with the initial E bit selections */
            for(i=0; i < 48; i++)
                E[i] = (char) e2[i];
        }
        
        protected override void  printResult()
        {
            throw new NotImplementedException();
        }

        protected override bool  isValid()
        {
            return true;
        }
    }
}
