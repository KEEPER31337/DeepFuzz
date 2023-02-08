#import numpy as np
import os
import re
from subprocess import Popen, PIPE, STDOUT


def verify_correctness(text, filename, mode):
    name = filename[:-2]
    postfix = filename[-2:]
    filename = name + '_' + mode + postfix
    f = open(filename, 'w')
    command = 'gcc -c -w ' + filename
    f.write(text)
    p1 = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    stderr = str(p1.stdout.read())
    print(stderr)
    if ('internal compiler error' in stderr):
        return True
    if ('error' in stderr):
        return False
    else:
        return True

def remove_comment(text):
    pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE)
    
    text_find = pattern.findall(text)

    for t in text_find:
        if t.startswith('/'):
            text = re.sub(pattern," ",t)

    return text 
 


# 여기서 매개변수가 없는 매크로와 매개변수가 있는 매크로에서 이름이 겹치면 변형됨 
# 때문에 매개변수 있는 것을 먼저 바꾸고 후에 매개변수 없는 것을 바꿈
# 주석처리 된 코드를 일단 사용하지 않기로 생각을 해서 parameter에서 filename을 제외함 
def replace_macro(text):

    '''postfix = filename[-2:]
    tempfile = 'temp' + postfix
    print(text, file=open(tempfile, 'w'))
    command = 'gcc -E ' + tempfile + ' > temp'
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    text = open('temp', 'r').read()'''

    ### 매크로 정보 추출하기
    macros = re.findall(r'#define .*\n', text)

    macro_remove_define = []
    for m in macros:
        macro_remove_define.append(re.sub(r'#define ','',m))


    macro_name = []
    macro_content = []
    macro_function = []
    macro_function_name = []
    macro_function_content = []
    macro_parameters = []

    for m in macro_remove_define:

        if ('(' and ')') in m: ### 매개변수가 있는 경우
            macro_function.append(m.split(' ')[0])
            macro_function_content.append(m.split(' ')[1].rstrip())
        else: ### 매개변수가 없는 경우
            macro_name.append(m.split(' ')[0])
            macro_content.append(m.split(' ')[1].rstrip())

    
    text = re.sub(r'#define .*\n', '', text) # 제일 위에 정의된 #define 지우기

    for f in macro_function:
        macro_function_name.append(f.split('(')[0])
        macro_parameters.append(f.split('(')[1].lstrip('(').rstrip(')').split(','))

    ############## 매개변수가 있는 경우
    temp_list=[]
    real_functions = []
    real_parameters=[]
    real_content = []
    for n in macro_function_name:
        ### (로 시작하고 \n 으로 끝나는 것 받아오기
        temp_list.append(re.findall(n+r'\(.*\n',text))

    ### parameter 뽑기
    for t in temp_list:
        temp = re.sub(r'.*\(','',''.join(t))
        real_parameters.append(re.sub(r'\).*','',temp).rstrip('\n').split(','))

    ### macro함수가 func(para1, para2, ...)이런 식으로 작성되어있다는 가정 하에 지우는 기능이다.
    ### func(para1,para2) 혹은 func(para1,    para2) 이런거 말고
    for n in macro_function_name:
        ind = macro_function_name.index(n)
        m_func = n+'('+','.join(real_parameters[ind])+')'
        real_functions.append(m_func)

    ### macro 이름을 함수로 바꾸기
    for content in macro_function_content:
        ind = macro_function_content.index(content)
        
        for i in range(len(macro_parameters[ind])): 
            content = re.sub(macro_parameters[ind][i],real_parameters[ind][i],content)
        
        real_content.append(content)
                
    ### macro 함수 content로 대체하기
    for f in real_functions:
        ind = real_functions.index(f)
        if f in text:
            ### re.sub는 정규표현식이 있어야만 작동을 한다. 따라서 여기서는 정규표현식이 없으므로 replace를 사용함
            text = text.replace(f,real_content[ind])

    ##############3 매개변수 없는 경우
    for m in macro_name: # code에 있는 macro 이름을 macro 내용으로 바꾸기 
        ind = macro_name.index(m)
        if m in text:
            text = re.sub(m,macro_content[ind],text)

    return text

def remove_space(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s', ' ', text)
    text = re.sub(r'\\t', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text