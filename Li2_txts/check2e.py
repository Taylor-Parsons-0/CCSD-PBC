import numpy as np
import sys
import os

log=sys.argv[1]

with open("occ.txt","r") as reader:
	text=[]
	for line in reader:
		text.append(line.split())
NB=len(text[1])

with open(f"{log}","r") as reader:
	textl=[]
	texts=[]
	for line in reader:
		textl.append(line)
		texts.append(line.split())

N1=0
es1=np.zeros((NB,NB,NB,NB))
for i in range(len(texts)):
	if "I=" and "J=" and "K=" and "L=" in textl[i] and len(texts[i])>4:
		N1+=1
		val=float(texts[i][9].replace("D","E"))
		es1[int(texts[i][1])-1,int(texts[i][3])-1,int(texts[i][5])-1,int(texts[i][7])-1]=val
N2=0
es2=np.zeros((NB,NB,NB,NB))
with open("twoeint.txt","r") as reader:
	text=[]
	for line in reader:
		text.append(line.split())
del text[0]
for i in range(len(text)):
	if "I=" and "J=" and "K=" and "L=" in text[i] and len(text[i])>4:
		N2+=1
		val=float(text[i][9].replace("D","E"))
		es2[int(text[i][1])-1,int(text[i][3])-1,int(text[i][5])-1,int(text[i][7])-1]=val
print(np.sum(abs(es1-es2)))
print(N1)
print(N2)
