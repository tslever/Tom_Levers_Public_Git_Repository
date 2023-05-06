
// Allow creating a vector of filenames of training images.
#include <vector>

// Allow creating elements of vector.
// Allow converting strings to C strings.
#include <string>

// Allow creating containerForFileInformation and searchHandle.
// Allow using FindFirstFileA and FindNextFileA.
#include <windows.h>


// Define method GetFilenamesOfTrainingImages.
std::vector<std::string> GetFilenamesOfEncodingsForTrainingImages(std::string path)
{
    // Declare a vector of filenames.
    std::vector<std::string> filenames;

    // Define a search expression for training images.
	std::string searchExpression = path + "*.txt";

    // Define a container  for file information about a found file.
	WIN32_FIND_DATAA containerForFileInformation;

    // Store information about the first JPG file found in containerForFileInformation, and
    // create a search handle that will be used in a subsequent call to FindNextFile.
	HANDLE searchHandle = FindFirstFileA(searchExpression.c_str(), &containerForFileInformation);

    // On the first call to do,
    // copy the filename of the first file found from containerForFileInformation
    // to a new element at the end of filenames.
    // Then,
    // search for another file and update searchHandle and containerForFileInformation.
    // If a file was found, copy the filename to filenames.
	do {
		filenames.push_back(containerForFileInformation.cFileName);
	} while (FindNextFileA(searchHandle, &containerForFileInformation));

    // Return filenames.
    return filenames;

} // GetFilenamesOfTrainingImages.