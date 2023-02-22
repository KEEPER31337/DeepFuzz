import subprocess
from os import sys
import datetime
import re

def getgccresult(srcPath,resPath):
     files = []
     for n in range(5):
          command = "gcc " + "-O" + n
          running = subprocess.Popen(["strace", "-e", "write", "-f", command, srcPath],
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
          stderr = running.stdout.read()

          d = datetime.datetime.now()
          d1 = re.sub(r"[-]","", str(d))
          filename = "output00" + n + ":" + re.sub(r"[-]","", str(d1[2:14]))
          file = resPath + filename + ".txt"
          files.append(file)
          sys.stdout = open(file,"w")
          print(stderr)
          sys.stdout.close()

     return files