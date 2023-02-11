import analyzer
import output
import generate as g
import preprocess
import train as t
import prepare as p
import autocompile

if __name__ == '__main__':
    print("Deep Fuzz Automation Framework")
    seedPath = input("Enter the seed path: ")
    dataPath = input("Enter the data path: ")

    p.prepare(seedPath)

    t.train(dataPath)

    g.generate(seedPath, dataPath)