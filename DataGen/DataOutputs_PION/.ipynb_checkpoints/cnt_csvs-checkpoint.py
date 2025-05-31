import pandas as pd
with open('sim_counts.csv','w') as f:
	for i in range(1,10001):
		filename = 'job' + str(i) + '.csv'
		try:
			rowcount = 0
			with open(filename) as fi:
				for row in fi:
					rowcount += 1
			f.write(str(i) + ',' + str(rowcount) + '\n')	
		except FileNotFoundError:
			f.write(str(i) + ',' + 'nan\n')

