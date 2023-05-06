
# SOURCE_FILES must be in format [FILENAME][SPACE][FILENAME][SPACE][FILENAME]...
# [FILENAME] cannot be in quotes.
SOURCE_FILES = ConvolveAudio.cu ConvertIntDataToFloatData.cu PerformAugmentedConvolution.cu ConvertFloatDataToIntData.cu

# [FILENAME] cannot be in quotes.
HEADER_FILES = ConvertIntDataToFloatData.h PerformAugmentedConvolution.h ConvertFloatDataToIntData.h

# INCLUDE_DIRECTORIES must be in format [PATH],[PATH],[PATH]...
# [PATH] may be written without quotes if [PATH] does not contain a space.
# For uniformity, I have enclosed all paths in quotes.
INCLUDE_DIRECTORIES = "C:\Users\Tom\Documents\GitHub\Toms_Levers_Git_Repository\ConvolveAudio\ConvolveAudio"

# LIBRARIES_TO_LINK must be in the format [PARTIAL_PATH],[PARTIAL_PATH],[PARTIAL_PATH]...
# [PARTIAL_PATH] consists of path to directory of library and prefix of library filename.
# [PARTIAL_PATH] may be written without quotes.
# For uniformity, I have enclosed all partial paths in quotes.
# [PARTIAL_PATH] cannot have spaces.
#LIBRARIES_TO_LINK =

# PREFIX_OF_OUTPUT_FILE_NAME may be written without quotes.
# I do not write filenames with spaces.
PREFIX_OF_NAME_OF_OUTPUT_FILE = "ConvolveAudio"

# Make output file whenever a source file or a header file changes.
# Use the nvcc compiler.
# Output severe, significant, production-quality, and informational compiler warnings.
# Reference include directories (i.e., directories of header files).
# Reference directories of libraries.
# Specify the prefix of the name of the output file.
makeTrainConvolutionalNeuralNetwork: $(SOURCE_FILES) $(HEADER_FILES)
	nvcc --compiler-options -W4 $(SOURCE_FILES) -I $(INCLUDE_DIRECTORIES) -o $(PREFIX_OF_NAME_OF_OUTPUT_FILE)