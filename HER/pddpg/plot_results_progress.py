import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
from ipdb import set_trace

if __name__ == "__main__":
	dirname = argv[1]

	print("checking the dir %s"%dirname)
	try:
		data = pd.read_csv(dirname + "progress.csv")
		data = data.fillna(0.0)
		print(data["eval/success"][-10:])
		plt.subplot(2,1,1)
		plt.plot(data["total/epochs"], data["eval/return_history"], label='eval')
		plt.plot(data["total/epochs"], data["rollout/return_history"], label='train')
		plt.legend()
		plt.xlabel('Epochs --->')
		plt.ylabel('Episode Reward ---->')
		
		plt.subplot(2,1,2)
		plt.plot(data["total/epochs"], data["eval/success"], label='eval')
		plt.legend()
		plt.xlabel('Epochs --->')
		plt.ylabel('Success ---->')
	
		plt.show()

		# print("last reward in train:",data["rollout/return_history"][-1])

	except Exception as e:
		print(e)

