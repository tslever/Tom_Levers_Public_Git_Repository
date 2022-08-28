
On Windows 10 64-bit, as administrator,

Install GeForce Game Ready Driver 461.72 - WHQL: https://www.nvidia.com/en-us/drivers/results/170887/

Install Visual Studio Community 2019.

Install CUDA 11.3.0.

Install NVIDIA Nsight 2021.1.0.

Install OpenCV-4.5.2-vc14-vc15: https://sourceforge.net/projects/opencvlibrary/

Install cuDNN 7.6.5: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_765/cudnn-install/index.html

Install chocolatey

Install make via chocolatey: choco install make

Ensure system PATH has:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\libnvvp
C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common
C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR
C:\ProgramData\chocolatey\bin
C:\Program Files\NVIDIA Corporation\Nsight Compute 2021.1.0\
C:\opencv_4.5\build\x64\vc15\bin
C:\Program Files\dotnet\
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64 (Allows nvcc to find cl.exe)
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\lib\x64 (see Makefile for justification)

Run make