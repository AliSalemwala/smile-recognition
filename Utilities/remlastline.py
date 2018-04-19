import os, sys

for file in os.listdir (sys.argv[1]):
	with open(sys.argv[1] + '\\' + file, 'r+') as f:
		lines = f.read()
		f.seek(0)
		f.write (lines[:-1])
		f.truncate()