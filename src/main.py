import analyzer as a
import output as o
import generate as g
import preprocess
import train as t
import prepare as p
import autocompile as ac
import os

if __name__ == '__main__':
    print("Deep Fuzz Automation Framework")
    seedPath = input("Enter the seed path: ")
    dataPath = input("Enter the data path: ")

    p.prepare(seedPath)
    t.train(dataPath)
    valid_files = g.generate(seedPath, dataPath)

    parsing_results = []
    for file in valid_files:

        resPath = os.path.dirname(file) # 파일의 폴더 경로
        compileResultFilenames = ac.getgccresult(file, resPath) # file 5개

        for f in compileResultFilenames: 

            contents = o.readFileContents(f)
            if(o.isError(contents)):
                errorLines = o.storeErrorLine(contents)
                option = print('Input the option: ')
                data = o.makeJsonObject(errorLines, f, option)

                o.makeJsonFile(file,data)
            
            print(a.parsing_json(f))
            parsing_results.append(a.parsing_json(file))