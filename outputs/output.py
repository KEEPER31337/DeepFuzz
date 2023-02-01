import json
    
def classifyErrorMsg(compiler, fileName):
    with open(fileName, "r") as file:
        contents = [content.rstrip() for content in file]

    if compiler == "gcc":
        errorLines = storeErrorLine(contents)
        inputName = setInputName(contents)
        data = makeJsonObject(errorLines, inputName)
        
    makeJsonFile(fileName, data)

def setInputName(contents):
    strings = contents[0].split(':')
    return strings[0]

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

def makeJsonObject(errorLines, inputName):
    data = {
        "errorMsg": []
    }

    if errorLines["syntaxError"]:
        data["errorMsg"].append({
            "compiler": "gcc",
            "input": inputName,
            "output": "syntaxError",
            "lines": errorLines["syntaxError"],
            "option": "-O3"
        })

    if errorLines["internalError"]:
        data["errorMsg"].append({
            "compiler": "gcc",
            "input": inputName,
            "output": "internalError",
            "lines": errorLines["internalError"],
            "option": "-O3"
        })

    return data

def makeJsonFile(fileName, data):
    jsonFile = fileName[:-4] + ".json"
    json_object = json.dumps(data, indent=4)
    with open(jsonFile, "w") as outfile:
        outfile.write(json_object)

classifyErrorMsg("gcc", "/Users/leesoobeen/Desktop/DeepFuzz/outputs/output001:2301182216.txt")