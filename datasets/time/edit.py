fOut = open('TIME_clean.QUE','w')
fIn = open('TIME_clean1.QUE','r')
for line in fIn:
    if '.I' in line or '.W' in line: 
        fOut.write(line)
    else:
        fOut.write(line.lower())
fOut.close()
fIn.close()
