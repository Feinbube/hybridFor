'''
This program adds the file header specified in the textPath variable 
to all files in the folder and subfolders specified by the path variable 
which have the the matching fileExtension
@author: Ralf
'''

import os

textPath = "D:\\Studium\\Arbeit\\ParProg\\Hybrid\\trunk\\License\\headerJanArne.txt"
path = "D:\\Studium\\Arbeit\\ParProg\\Hybrid\\trunk\\Hybrid\\MSilToOpenCL"
fileExtension = "cs"

def preheader(fileName):
    return "/*    \n*    " + fileName + "\n*\n";

def writeHeaderToAllFilesInDir():
    headerHandle = open(textPath, "r")
    header = headerHandle.read()
    headerHandle.close()
    writeHeaderToEveryFileInDir(header, path)
    

def writeHeaderToEveryFileInDir(header, path):
    for root, dirs, files in os.walk(path):
        for fileName in files:
            fileParts=fileName.split(".")
            if fileParts[-1] == fileExtension :
                fileHandle = open(root + "\\" + str(fileName), "r")
                tempFile = fileHandle.read()
                fileHandle.close()
                
                fileHandle = open(root + "\\" + str(fileName), "w")
                fileHandle.write(preheader(fileName))
                fileHandle.write(header);
                fileHandle.write(tempFile)
                fileHandle.close()


if __name__ == '__main__':
    writeHeaderToAllFilesInDir()
