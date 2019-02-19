fOut = open('query-text_clean','w')
fIn = open('query-text','r')
for line in fIn:
    if '.I' in line or '.W' in line: 
        fOut.write(line)
    else:
        fOut.write(line.lower())
fOut.close()
fIn.close()
