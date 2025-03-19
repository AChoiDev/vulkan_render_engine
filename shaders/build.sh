#!/bin/bash

glslc_cmd="C:\VulkanSDK\1.3.280.0\Bin\glslc.exe"

# Get the current directory of the script
currentDirectory=$(pwd)

# Define the source directory relative to the current directory
sourceDirectory="$(pwd)/shaders/src"

# Define the destination directory relative to the current directory
destinationDirectory="$(pwd)/shaders/spirv"

# Create the destination directory if it doesn't exist
# mkdir -p "$destinationDirectory"

# Find all files with .spv extension in the source directory
fileToCompile=$(find "$sourceDirectory" -type f)

# Loop through each file found
for file in $fileToCompile; do
    # echo $file
    # # Get the base filename without the extension
    baseName=$(basename "$file" .glsl)
    outName="$baseName.spv"
    # echo $outName
    # echo $file

    # Compile the shader
    $glslc_cmd "$file" -o "$destinationDirectory/$outName"
    echo "Compiled  $destinationDirectory/$outName"
done
