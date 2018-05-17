fOut = open('cran.notitle.all.1400','w')
fIn = open('cran.all.1400','r')

inT = False
inW = False
for line in fIn:

    if inT:
        titleLines += 1

    if '.T' in line:
        inT = True
    elif '.A' in line:
        inT = False
    elif '.W' in line:
        inW = True
        fOut.write('.W\n')
    elif '.I' in line:
        inW = False
        titleLines = 0

    if inW:
        if titleLines >= 1:
            titleLines -= 1
        else:
            fOut.write(line)
    else:
        fOut.write(line)

fOut.close()
fIn.close()
