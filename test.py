# Tensorflow testing for predicting if given values
# are for weekend days or not, based on number of events 
# and hour of the day. 
#
# Predictions are done based kNN (nearest neighbour)
# algorithm which may or may not be best choice
# for this...
#
# Learning dataset: input/data.csv
# Testing dataset: test/data.csv

import csv
import tensorflow

class TensorflowTest:
	input_data = []
	input_data_y = []
	test_data = []

	def __init__(self):
		self.read_input_data()
		self.read_test_data()
		self.run_test() # Create prediction function for tensor

	def read_input_data(self):
		# As input we read number of events, time of the day
		# and what we're looking to predict (actual day) to 
		# separate index (day number) which is what we're looking for 
		# as an actual answer.
		#
		# CSV format: Events,Hour,Day
		with open("input/data.csv","r") as file:
			reader = csv.reader(file)
			for row in reader:
				self.input_data.append([row[0],row[1]])
				# Days index, we will attempt to guess
				# most likey day from here
				self.input_data_y.append(row[2])

	def read_test_data(self):
		# Testing data does not include the answer (day number).
		# CSV format: Events,Hour
		with open("test/data.csv","r") as file:
			reader = csv.reader(file)
			for row in reader:
				self.test_data.append([row[0],row[1]])
		
	def run_test(self):
		# Input and tested value will be fed to these
		input_tensor = tensorflow.placeholder("float",[None,len(self.input_data[0])])
		compare_tensor = tensorflow.placeholder("float",[len(self.input_data[0])])

		# Find for nearest match in data set
		# (kNN Formula found somewhere in the documentation :P)
		nearest_match = tensorflow.reduce_sum(
			tensorflow.abs(
				tensorflow.add(
					input_tensor,
					tensorflow.negative(compare_tensor)
				)
			), 
			axis=1 # Reduce to 1 dimensional tensor
		)

		# Lookup function for Tensor session which is used for each guess
		lookup_function = tensorflow.argmin(nearest_match, 0)

		with tensorflow.Session() as session:
			session.run(tensorflow.global_variables_initializer())
			for i in range (len(self.test_data)):
				test_row = self.test_data[i]

				# Tensorflow session attempting to guess correct day for the values
				likely_day = session.run(
					lookup_function,
					feed_dict={
						input_tensor:self.input_data,
						compare_tensor:self.test_data[i]
					}
				)

				if likely_day < 6:
					print(f"Day with {test_row[0]} events at hour {test_row[1]} is likely a week day")
				else:
					print(f"Day with {test_row[0]} events at hour {test_row[1]} is likely a weekend")


# Run the test
t = TensorflowTest()

