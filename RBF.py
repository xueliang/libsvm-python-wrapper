

import sys
import os
import re
from subprocess import *

# svm, grid, and gnuplot executable files

is_win32 = (sys.platform == 'win32')
if not is_win32:
	svmtrain_exe = "/tools/libsvm-3.11/svm-train"
	svmpredict_exe = "/tools/libsvm-3.11/svm-predict"
	svmscale_exe  = "/tools/libsvm-3.11/svm-scale"
	gnuplot_exe = r'gnuplot_exe'
	grid_py = "/tools/libsvm-3.11/mytools/grid.py"
else:
	svmtrain_exe = r"U:\\tools\\libsvm-3.11\\windows\\svm-train.exe"
	svmpredict_exe = r"U:\\tools\\libsvm-3.11\\windows\svm-predict.exe"
	svmscale_exe  = "U:\\tools\\libsvm-3.11\\windows\svm-scale.exe"
	gnuplot_exe = r'gnuplot_exe'
	grid_py = r"U:\\tools\\libsvm-3.11\\mytools\\grid.py"

class svmRBF():
    def __init__(self):
        assert os.path.exists(svmtrain_exe),"svm-train executable not found"
        assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
        assert os.path.exists(grid_py),"grid.py not found"
        
    def scale(self,inputfile,scaledfile=None,rangefile=None):
        if scaledfile == None:
            scaledfile = inputfile + '.scaled'
        if rangefile == None:
            rangefile = inputfile + '.range'
        cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, rangefile, inputfile, scaledfile)
        print('Scaling data...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()
        
    def scalewithrange(self,inputfile,scaledfile,rangefile):
        if scaledfile == None:
            scaledfile = inputfile + '.scaled'
        cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, rangefile, inputfile, scaledfile)
        print('Scaling data...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()
        
    def gridsearch(self,trainingfile):
        cmd = 'python {0} -svmtrain "{1}" "{2}"'.format(grid_py, svmtrain_exe, trainingfile)
        print('Cross validation...')
        f = Popen(cmd, shell = True, stdout = PIPE).stdout
        line = ''
        while True:
            last_line = line
            line = f.readline()
            if not line: break
        c,g,rate = map(float,last_line.split())
        return c,g,rate
    
    def train(self,c,g,trainingfile,modelfile=None):
        if modelfile == None:
            modelfile = trainingfile + '.model'
        cmd = '{0} -c {1} -g {2} "{3}" "{4}"'.format(svmtrain_exe,c,g,trainingfile,modelfile)
        print('Training...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()
        
    def predict(self,testingfile,modelfile,predictfile = None):
        if predictfile == None:
            predictfile = testingfile + '.predict'
        cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, testingfile, modelfile, predictfile)
        print('Testing...')
        f = Popen(cmd, shell = True,stdout= PIPE).stdout
        line = ''
        while True:
            last_line = line
            line = f.readline()
            if not line: break
        accu = re.findall('Accuracy = (100|[0-9]{1,2}|[0-9]{1,2}\.[0-9]+)%.*',last_line)[0]
        return float(accu)

if __name__ == '__main__':
    trainingfile = 'splice.txt'
    rbf = svmRBF()
    rangefile = trainingfile + '.range'
    scaledtraining = trainingfile + '.scale'
    rbf.scale(trainingfile,scaledtraining,rangefile)

    testingfile = 'splice.t'
    scaledtesting = testingfile + '.scale'
    rbf.scalewithrange(testingfile,scaledtesting,rangefile)
    
    c,g,r = rbf.gridsearch(scaledtraining)
    print "best c,g is", c,g
    modelfile = trainingfile + '.model'
    rbf.train(c,g,scaledtraining,modelfile)
    
    accutest = rbf.predict(scaledtraining,modelfile)
    accutrain = rbf.predict(scaledtesting,modelfile)
    print "different on gridsearch, trainingdata and testing is", r, accutrain, accutest
    
    
    