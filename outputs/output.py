import json

def readFileContents(fileName):
    with open(fileName, "r") as file:
        contents = [content.rstrip() for content in file]
    
    return contents

def isError(contents):
    error = False
    for content in contents:
        try:
            strings = content.split(":")
            if "error" in strings[3]:
                error = True
                break
        except:
            return error
    
    return error

def storeErrorLine(contents):
    errorLines = { "syntaxError": [], "internalError": [] }

    for content in contents:
        strings = content.split(':')
        if strings[3] == " error" or strings[3] == " fatal error":
            errorLines["syntaxError"].append(strings[1])
        elif strings[3] == " internal compiler error":
            errorLines["internalError"].append(strings[1])

    errorLines["syntaxError"] = list(set(errorLines["syntaxError"]))
    errorLines["internalError"] = list(set(errorLines["internalError"]))

    return errorLines

def makeJsonObject(errorLines, fileName, option):
    data = {
        "errorMsg": []
    }

    if errorLines["syntaxError"]:
        data["errorMsg"].append({
            "compiler": "gcc",
            "input": fileName,
            "output": "syntaxError",
            "lines": errorLines["syntaxError"],
            "option": option
        })

    if errorLines["internalError"]:
        data["errorMsg"].append({
            "compiler": "gcc",
            "input": fileName,
            "output": "internalError",
            "lines": errorLines["internalError"],
            "option": option
        })

    return data

def makeJsonFile(outputName, data):
    jsonFile = outputName[:-4] + ".json"
    json_object = json.dumps(data, indent=4)
    with open(jsonFile, "w") as outfile:
        outfile.write(json_object)
