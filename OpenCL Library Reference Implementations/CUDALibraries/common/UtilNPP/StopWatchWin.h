#ifndef NV_NPP_UTIL_STOP_WATCH_WIN_H
#define NV_NPP_UTIL_STOP_WATCH_WIN_H
/*
* Copyright 2008-2009 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

// includes, system
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

namespace npp
{

    /// Windows specific implementation of StopWatch
    class StopWatchWin 
    {
    protected:

        //! Constructor, default
        StopWatchWin();

        // Destructor
        ~StopWatchWin();

    public:

        //! Start time measurement
        inline void start();

        //! Stop time measurement
        inline void stop();

        //! Reset time counters to zero
        inline void reset();

        //! Time in msec. after start. If the stop watch is still running (i.e. there
        //! was no call to stop()) then the elapsed time is returned, otherwise the
        //! time between the last start() and stop call is returned
        inline const double elapsed() const;


    private:

        // member variables

        //! Start of measurement
        LARGE_INTEGER  start_time;
        //! End of measurement
        LARGE_INTEGER  end_time;

        //! Time difference between the last start and stop
        double  diff_time;

        //! TOTAL time difference between starts and stops
        double  total_time;

        //! flag if the stop watch is running
        bool running;

        //! tick frequency
        static double  freq;

        //! flag if the frequency has been set
        static  bool  freq_set;
    };

    // functions, inlined

    ////////////////////////////////////////////////////////////////////////////////
    //! Start time measurement
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchWin::start() 
    {
        QueryPerformanceCounter((LARGE_INTEGER*) &start_time);
        running = true;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Stop time measurement and increment add to the current diff_time summation
    //! variable. Also increment the number of times this clock has been run.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchWin::stop() 
    {
        QueryPerformanceCounter((LARGE_INTEGER*) &end_time);
        diff_time = (float) 
            (((double) end_time.QuadPart - (double) start_time.QuadPart) / freq);

        total_time += diff_time;
        running = false;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Reset the timer to 0. Does not change the timer running state but does 
    //! recapture this point in time as the current start time if it is running.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchWin::reset() 
    {
        diff_time = 0;
        total_time = 0;
        if( running )
            QueryPerformanceCounter((LARGE_INTEGER*) &start_time);
    }


    ////////////////////////////////////////////////////////////////////////////////
    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned added to the 
    //! current diff_time sum, otherwise the current summed time difference alone
    //! is returned.
    ////////////////////////////////////////////////////////////////////////////////
    inline const double
    StopWatchWin::elapsed() const 
    {
        // Return the TOTAL time to date
        double retval = total_time;
        if(running) 
        {
            LARGE_INTEGER temp;
            QueryPerformanceCounter((LARGE_INTEGER*) &temp);
            retval +=  
                (((double) (temp.QuadPart - start_time.QuadPart)) / freq);
        }

        return retval;
    }

} // npp namespace

#endif // NV_NPP_UTIL_STOP_WATCH_WIN_H

