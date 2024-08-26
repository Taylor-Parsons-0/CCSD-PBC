import numpy as np

with open("out.txt","r") as reader:
	text=[]
	for line in reader:
		text.append(line.split())
del text[0]
del text[0]
del text[0]
del text[0]

print(text)
