f = open("full_text_sentences_new.txt")
lines = f.readlines()

m = {}

for line in lines:
	line = line.strip()

	for w in line.split(" "):
		if w not in m: m[w] = 0
		m[w] += 1

while 1:
    word = raw_input("Word> ").strip()
    if word not in m: print "freq = 0"
    else: print "freq =", m[word]