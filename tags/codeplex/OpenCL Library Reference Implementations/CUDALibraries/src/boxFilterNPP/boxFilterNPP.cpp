/*
 * Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
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

// Sources:
 // [1] http://developer.nvidia.com/cuda-libraries-sdk-code-samples --> Box Filter with NPP


//Usage:
// Command line parameters:
	// x : width of the image
	// y : height of the image
	// w : number of warmup rounds
	// m : number of measured rounds
	// c : number of cooldown rounds
	
	// example: -m=10 => 10 rounds with measured time are executed
	
	// -print enables output print


#include <npp.h>

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>


#include <string.h>
#include <fstream>
#include <iostream>


#include <shrQATest.h>
#include <cutil_inline.h>

#ifdef WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif

bool g_bQATest = false;
int  g_nDevice = -1;

size_t imgX = 1024;
size_t imgY = 1024;
Npp8u * srcImgData;
Npp8u * dstImgData;

bool printSamplePixels = false;
int numWarmupRounds = 5;
int numMeasuredRounds = 3;
int numCoolDownRounds = 2;


void processCommandLine(int argc, char **argv)
{
	int cmdVali = 0;
	if (cutGetCmdLineArgumenti( argc, (const char**)argv, "x", &cmdVali))
	{
		imgX = cmdVali;
		printf("The number of pixels in x-direction is set to: %d\n", imgX);
	}
	if (cutGetCmdLineArgumenti( argc, (const char**)argv, "y", &cmdVali))
	{
		imgY = cmdVali;
		printf("The number of pixels in y-direction is set to: %d\n", imgY);
	}

	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "w", &cmdVali))
	{
		numWarmupRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numWarmupRounds);
	}
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "m", &cmdVali))
	{
		numMeasuredRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numMeasuredRounds);
	} 
	if (cutGetCmdLineArgumenti(argc, (const char**)argv, "c", &cmdVali))
	{
		numCoolDownRounds = cmdVali;
		printf("The number of warm-up rounds is set to: %d\n", numCoolDownRounds);
	}

	if (cutCheckCmdLineFlag( argc, (const char**)argv, "print"))
	{
		printSamplePixels = true;
		printf("Printing is enabled\n");
	}
}




inline void cudaSafeCallNoSync( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        std::cerr << file << "(" << line << ")" << " : cudaSafeCallNoSync() Runtime API error : ";
	std::cerr << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

inline int cudaDeviceInit()
{
    int deviceCount;
    cudaSafeCallNoSync(cudaGetDeviceCount(&deviceCount), __FILE__, __LINE__);
    if (deviceCount == 0) {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(-1);
    }
    int dev = g_nDevice;
    if (dev < 0) 
        dev = 0;
    if (dev > deviceCount-1) {
        std::cerr << std::endl << ">> %d CUDA capable GPU device(s) detected. <<" << deviceCount << std::endl;
        std::cerr <<">> cutilDeviceInit (-device=" << dev << ") is not a valid GPU device. <<" << std::endl << std::endl;
        return -dev;
    }  else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl; 
    }
    cudaSafeCallNoSync(cudaSetDevice(dev), __FILE__, __LINE__);

    return dev;
}

void parseCommandLineArguments(int argc, char *argv[])
{
    if (argc >= 2) {
       for (int i=1; i < argc; i++) {
            if (!STRCASECMP(argv[i], "-qatest")   || !STRCASECMP(argv[i], "--qatest") ||
                !STRCASECMP(argv[i], "-noprompt") || !STRCASECMP(argv[i], "--noprompt")) 
            {
               g_bQATest = true;
            }

            if (!STRNCASECMP(argv[i], "-device", 7)) {
               g_nDevice = atoi(&argv[i][8]);
            } else if (!STRNCASECMP(argv[i], "--device", 8)) {
               g_nDevice = atoi(&argv[i][9]);
            }
            if (g_nDevice != -1) {
               cudaDeviceInit();
            }
      }
   }
}

void evaluateErrorCode(NppStatus status)
{
	switch(status)
	{
	case NPP_NOT_SUPPORTED_MODE_ERROR:
		printf("NPP_NOT_SUPPORTED_MODE_ERROR");
		break;
	case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
		printf("NPP_ROUND_MODE_NOT_SUPPORTED_ERROR");
		break;
	case NPP_RESIZE_NO_OPERATION_ERROR:
		printf("NPP_RESIZE_NO_OPERATION_ERROR");
		break;
	case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
		printf("NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY");
		break;
	case NPP_BAD_ARG_ERROR:
		printf("NPP_BAD_ARG_ERROR");
		break;
	case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
		printf("NPP_LUT_NUMBER_OF_LEVELS_ERROR");
		break;
	case NPP_TEXTURE_BIND_ERROR:
		printf("NPP_TEXTURE_BIND_ERROR");
		break;
	case NPP_COEFF_ERROR:
		printf("NPP_COEFF_ERROR");
		break;
	case NPP_RECT_ERROR:
		printf("NPP_RECT_ERROR");
		break;
	case NPP_QUAD_ERROR:
		printf("NPP_QUAD_ERROR");
		break;
	case NPP_WRONG_INTERSECTION_ROI_ERROR:
		printf("NPP_WRONG_INTERSECTION_ROI_ERROR");
		break;
	case NPP_NOT_EVEN_STEP_ERROR:
		printf("NPP_NOT_EVEN_STEP_ERROR");
		break;
	case NPP_INTERPOLATION_ERROR:
		printf("NPP_INTERPOLATION_ERROR");
		break;
	case NPP_RESIZE_FACTOR_ERROR:
		printf("NPP_RESIZE_FACTOR_ERROR");
		break;
	case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
		printf("NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR");
		break;
	case NPP_MEMFREE_ERR:
		printf("NPP_MEMFREE_ERR");
		break;
	case NPP_MEMSET_ERR:
		printf("NPP_MEMSET_ERR");
		break;
	case NPP_MEMCPY_ERROR:
		printf("NPP_MEMCPY_ERROR");
		break;
	case NPP_MEM_ALLOC_ERR:
		printf("NPP_MEM_ALLOC_ERR");
		break;
	case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
		printf("NPP_HISTO_NUMBER_OF_LEVELS_ERROR");
		break;
	case NPP_MIRROR_FLIP_ERR:
		printf("NPP_MIRROR_FLIP_ERR");
		break;
	case NPP_INVALID_INPUT:
		printf("NPP_INVALID_INPUT");
		break;
	case NPP_ALIGNMENT_ERROR:
		printf("NPP_ALIGNMENT_ERROR");
		break;
	case NPP_STEP_ERROR:
		printf("NPP_STEP_ERROR");
		break;
	case NPP_SIZE_ERROR:
		printf("NPP_SIZE_ERROR");
		break;
	case NPP_POINTER_ERROR:
		printf("NPP_POINTER_ERROR");
		break;
	case NPP_CUDA_KERNEL_EXECUTION_ERROR:
		printf("NPP_CUDA_KERNEL_EXECUTION_ERROR");
		break;
	case NPP_NOT_IMPLEMENTED_ERROR:
		printf("NPP_NOT_IMPLEMENTED_ERROR");
		break;
	case NPP_ERROR:
		printf("NPP_ERROR");
		break;
	case NPP_NO_ERROR:
		printf("NPP_NO_ERROR");
		break;
	case NPP_WARNING:
		printf("NPP_WARNING");
		break;
	case NPP_WRONG_INTERSECTION_QUAD_WARNING:
		printf("NPP_WRONG_INTERSECTION_QUAD_WARNING");
		break;
	case NPP_MISALIGNED_DST_ROI_WARNING:
		printf("NPP_MISALIGNED_DST_ROI_WARNING");
		break;
	case NPP_AFFINE_QUAD_INCORRECT_WARNING:
		printf("NPP_AFFINE_QUAD_INCORRECT_WARNING");
		break;
	case NPP_DOUBLE_SIZE_WARNING:
		printf("NPP_DOUBLE_SIZE_WARNING");
		break;
	case NPP_ODD_ROI_WARNING:
		printf("NPP_ODD_ROI_WARNING");
		break;
	default:
		printf("unknown error code: %d\n", status);
	}

}

void generateInput(Npp8u *inputImage, size_t imageX, size_t imageY)
{
	//srand((unsigned)time(0));
    for (size_t i = 0; i <  imageX * imageY; ++i)
    {
		inputImage[i] = rand();
    }
}

void printfNPPinfo(int argc, char *argv[])
{
    const char *sComputeCap[] = {
       "No CUDA Capable Device Found",
       "Compute 1.0", "Compute 1.1", "Compute 1.2",  "Compute 1.3",
       "Compute 2.0", "Compute 2.1", NULL
    };

    const NppLibraryVersion * libVer   = nppGetLibVersion();
    NppGpuComputeCapability computeCap = nppGetGpuComputeCapability();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);
    if (computeCap != 0 && g_nDevice == -1) {
        printf("Using GPU <%s> with %d SM(s) with", nppGetGpuName(), nppGetGpuNumSMs());
	if (computeCap > 0) {
           printf(" %s\n", sComputeCap[computeCap]);	
        } else {
           printf(" Unknown Compute Capabilities\n");
        }
    } else {
        printf("%s\n", sComputeCap[computeCap]);
    }
}

//note: no boundary checking!
void printPixel(size_t x, size_t y, Npp8u *src, size_t linePitchSrc, Npp8u *dst, size_t linePitchDst , int maskSizeX, int maskSizeY, int maskCenterX, int maskCenterY)
{
	printf("Pixel Print:\n");
	printf("On the left you can see the masked input image at pixel (%d,%d)\n",x,y);
	printf("On the right you can see the output image at pixel (%d,%d)\n", x-maskCenterX, y-maskCenterY);
	for(int i=0; i < maskSizeY; i++)
	{
		for(int j=0; j < maskSizeX; j++)
		{
			printf("%u ", src[x-maskCenterX+j + (y-maskCenterY+i) * linePitchSrc]);
		}
		if(i == maskCenterY)
		{
			printf("\t %u\n", dst[x-maskCenterX + (y-maskCenterY) * linePitchDst]); 
		}
		else
		{
			printf("\n");
		}
	}
}

void fillImage(npp::ImageCPU_8u_C1 *img, Npp8u *imgData)
{
	unsigned int nPitch = (*img).width();
    const Npp8u * pSrcLine = imgData;
    Npp8u * pDstLine = (*img).data();
    for (size_t iLine = 0; iLine < (*img).height(); ++iLine)
    {
		memcpy(pDstLine, pSrcLine, (*img).width() * sizeof(Npp8u));
        pSrcLine += nPitch;
        pDstLine += nPitch;
    }
}


int main(int argc, char* argv[])
{
    shrQAStart(argc, argv);

	processCommandLine(argc, argv);

	srcImgData = (Npp8u*) malloc(imgX*imgY*sizeof(Npp8u)); 
	generateInput(srcImgData, imgX, imgY);

    try
    {

        // Parse the command line arguments for proper configuration
        parseCommandLineArguments(argc, argv);

        printfNPPinfo(argc, argv);
                // declare a host image object for an 8-bit grayscale image
		npp::ImageCPU_8u_C1 oHostSrc(imgX, imgY);
				// fill image with random values
		fillImage(&oHostSrc, srcImgData);
                // declare a device image and copy construct from the host image,
                // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
                // create struct with box-filter mask size
        NppiSize oMaskSize = {3, 3};
                // create struct with ROI size given the current mask
        NppiSize oSizeROI = {oDeviceSrc.width() - oMaskSize.width + 1, oDeviceSrc.height() - oMaskSize.height + 1};
                // allocate device image of appropriatedly reduced size
        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
                // set anchor point inside the mask
        NppiPoint oAnchor = {1, 1};
                // run box filter
        NppStatus eStatusNPP;
		
		unsigned int timer = 0;
		cutCreateTimer(&timer);
		float overallTime = 0.0;
		
		for(int i = 0; i < numWarmupRounds; i++)
		{
			eStatusNPP = nppiFilterBox_8u_C1R(oDeviceSrc.data(oAnchor.x,oAnchor.y), oDeviceSrc.pitch(), 
											  oDeviceDst.data(), oDeviceDst.pitch(), 
											  oSizeROI, oMaskSize, oAnchor);
		}
		for(int i = 0; i < numMeasuredRounds; i++)
		{
			cutilDeviceSynchronize();
			cutResetTimer(timer);
			cutStartTimer(timer);
			eStatusNPP = nppiFilterBox_8u_C1R(oDeviceSrc.data(oAnchor.x,oAnchor.y), oDeviceSrc.pitch(), 
											  oDeviceDst.data(), oDeviceDst.pitch(), 
											  oSizeROI, oMaskSize, oAnchor);
			cutilDeviceSynchronize();
			// stop and destroy timer
			cutStopTimer(timer);
			float elapsedTime = cutGetTimerValue(timer);
			overallTime += elapsedTime;
			printf("The elapsed time for round %d is: %f ms.\n", i+1, elapsedTime);

		}
		for(int i = 0; i < numCoolDownRounds; i++)
		{
			eStatusNPP = nppiFilterBox_8u_C1R(oDeviceSrc.data(oAnchor.x,oAnchor.y), oDeviceSrc.pitch(), 
											  oDeviceDst.data(), oDeviceDst.pitch(), 
											  oSizeROI, oMaskSize, oAnchor);
		}
		
		overallTime /= numMeasuredRounds;	
		printf("The average elapsed time is: %f ms.\n\n", overallTime);
		cutDeleteTimer(timer);

        NPP_ASSERT(NPP_NO_ERROR == eStatusNPP);
                // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
                // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
        
		if (printSamplePixels)
		{	
			printPixel(10, 10, oHostSrc.data(), oHostSrc.pitch(), oHostDst.data(), oHostDst.pitch(), oMaskSize.width, oMaskSize.height, oAnchor.x, oAnchor.y);
			printPixel(100, 100, oHostSrc.data(), oHostSrc.pitch(), oHostDst.data(), oHostDst.pitch(), oMaskSize.width, oMaskSize.height, oAnchor.x, oAnchor.y);
			printPixel(200, 200, oHostSrc.data(), oHostSrc.pitch(), oHostDst.data(), oHostDst.pitch(), oMaskSize.width, oMaskSize.height, oAnchor.x, oAnchor.y);
		}

        shrQAFinish(argc, (const char **)argv, QA_PASSED);

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception & rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        shrQAFinish(argc, (const char **)argv, QA_FAILED);
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        shrQAFinish(argc, (const char **)argv, QA_FAILED);
        exit(EXIT_FAILURE);
        return -1;
    }
    
    return 0;
}