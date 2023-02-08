import json

def parsing_json(filename):
    with open(filename) as f:
        outJson = json.load(f)

    compiler = outJson['compiler']
    inputSource = outJson['input']
    compileResult = outJson['output']
    resultLines = outJson['lines']
    compileOption = outJson['option']

    if compileResult == "internalError":
        return [compiler, inputSource, resultLines, compileOption, 1]
    else:
        return [compiler, inputSource, resultLines, compileOption, -1]
 