import subprocess
from os import sys
import datetime
import re
def getgccresult(srcPath,resPath): 
     f =  open(srcPath,'wb')
     for n in range(5):
      command = "gcc"  + "-O" + n
      running = subprocess.Popen(["strace","-e", "write", "-f", command, srcPath],
                              shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      stderr = running.stdout.read()
     d = datetime.datetime.now()
     i = 0
     for i in range(n):
          if i == n:
               i == n+1
              
          filename = "output00" + i + ":" + re.sub(r"[-]","", d)
          sys.stdout = open(resPath + filename + ".txt",'w')
          print(stderr)
          sys.stdout.close()
     f.close()