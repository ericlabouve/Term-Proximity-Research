def toLowerCase():
	fOut = open('LISA.QUE','w')
	fIn = open('LISA1.QUE','r')
	lastWasI = False
	for line in fIn:
	    if '.I' in line: 
	        fOut.write(line)
	        lastWasI = True
	    elif lastWasI:
	    	fOut.write(line)
	    	lastWasI = False
	    else:
	        fOut.write(line.lower())
	fOut.close()
	fIn.close()

def removeTitles():
	fOut = open('lisa_clean_notitle.all','w')
	fIn = open('lisa_clean_notitle1.all','r')
	inTitle = False
	for line in fIn:
		if '.W' in line:
			fOut.write(line)
			inTitle = True
		elif inTitle:
			if len(line.strip()) == 0:
				inTitle = False
		else:
			fOut.write(line)

removeTitles()