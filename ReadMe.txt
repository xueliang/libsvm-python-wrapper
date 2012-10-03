
This package provide an interface to call libSVM in python easily. An example to use the packags is as follows:

	rbf = svmRBF()
	trainingfile = 'splice.txt'
	rangefile = trainingfile + '.range'
	scaledtraining = trainingfile + '.scale'
	print('Scaling training data...')
	rbf.scale(trainingfile,scaledtraining,rangefile)

	testingfile = 'splice.t'
	scaledtesting = testingfile + '.scale'
	print('Scaling testing data...')
	rbf.scalewithrange(testingfile,scaledtesting,rangefile)

	print('Cross validation...')
	c,g,r = rbf.gridsearch(scaledtraining)
	print "best c,g is", c,g
	
	print('Training...')
	modelfile = trainingfile + '.model'
	rbf.train(c,g,scaledtraining,modelfile)
	
	print('Testing...')
	accutest = rbf.predict(scaledtraining,modelfile)
	accutrain = rbf.predict(scaledtesting,modelfile)
	print "different on gridsearch, trainingdata and testing is", r, accutrain, accutest