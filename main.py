import csv
import numpy as np
import warnings
warnings.simplefilter('ignore')

def get_mean(a_vector):
	temp = 0
	for an_element in a_vector:
		temp = temp + an_element
	return temp/len(a_vector)

def get_deviation(mean, a_vector):
	temp = 0
	for an_element in a_vector:
		temp = temp + ((an_element - mean) ** 2)
	temp = temp/len(a_vector)
	return (temp ** (1/2))
	
# getting the total error
def get_total_error(data, parameter_vector):
	total_error = 0.0
	for a_vector in data:
		feature_vector = a_vector[1:14]
		prediction_value = a_vector[14]
		total_error += (prediction_value - feature_vector.dot(parameter_vector)) ** 2
	return total_error/400

# getting the ith term of the gradient
def get_kth_derivative(feature_vector, prediction_vector, parameter_vector, k):
	grad = 0.0
	for i in range(0, len(prediction_vector)):
		a_vector = feature_vector[i]
		prediction_value = prediction_vector[i]
		grad += (prediction_value - parameter_vector.dot(a_vector)) * (-2) * (a_vector[k])
	return grad/400

# getting the new w
def get_new_parameter_vector(data, parameter_vector, learning_rate):
	temp_vector = np.zeros(13)
	feature_vector = data[0:400, 1:14]
	prediction_vector = data[0:400, 14:15]
	for j in range(0,1000):
		for i in range(0, len(parameter_vector)):
			temp_vector[i] = parameter_vector[i] - (learning_rate * get_kth_derivative(feature_vector, prediction_vector, parameter_vector, i))
		parameter_vector = temp_vector
	return parameter_vector

# getting the pth norm of the vector
def get_p_norm(a_vector, p):
	norm = 0.0
	a_vector = np.absolute(a_vector)
	for an_element in a_vector:
		norm += an_element ** p
	return (norm ** (1/p))

# getting the pth norm error
def get_total_pnorm_error(data, parameter_vector, lambda_value, p):
	total_error = 400 * get_total_error(data, parameter_vector)
	total_error += (lambda_value * get_p_norm(parameter_vector, p))
	return total_error/400

# getting the ith term of the total gradient of the pth norm
def get_kth_pnorm_derivative(parameter_vector, lambda_value, p, k):
	temp = (parameter_vector[k])*(abs(parameter_vector[k]) ** (p-2))
	norm = get_p_norm(parameter_vector, p)
	# parameter_vector = np.absolute(parameter_vector)
	# for i in range(0, len(parameter_vector)):
	# 	temp += ((parameter_vector[i]) ** p)
	# temp = lambda_value * (temp ** ((1/p) - 1)) * ((parameter_vector[k]) ** (p-1))
	return temp/norm

def get_new_regularised_parameter_vector(data, parameter_vector, learning_rate, lambda_value, p, num_of_iterations):
	# feature_vector = a_vector[1:14]
	# prediction_vector = (feature_vector.dot(parameter_vector))
	temp_vector = np.zeros(13)
	feature_vector = data[0:400, 1:14]
	prediction_vector = data[0:400, 14:15]

	for j in range(0,num_of_iterations):
		for i in range(0, len(parameter_vector)):
			temp_vector[i] = parameter_vector[i] - (learning_rate * ((get_kth_derivative(feature_vector, prediction_vector, parameter_vector, i) + get_kth_pnorm_derivative(parameter_vector, lambda_value, p, i))))
		# if j%100 == 0:
		# 	print(get_total_error(data, parameter_vector), ",		", lambda_value * get_p_norm(parameter_vector, p), ",			", get_kth_derivative(feature_vector, prediction_vector, parameter_vector, i),",		", get_kth_pnorm_derivative(parameter_vector, lambda_value, p, i), ",		",(learning_rate * ((get_kth_derivative(feature_vector, prediction_vector, parameter_vector, i) + get_kth_pnorm_derivative(parameter_vector, lambda_value, p, i)))))
		parameter_vector = temp_vector
		# print(parameter_vector)
	return parameter_vector

# getting the predicted values
def get_prediction(test_data, parameter_vector, file_name):
	f = open(file_name, "a")
	f.write("ID,MEDV\n")
	i = 0;
	for a_vector in test_data:
		feature_vector = a_vector[1:14]
		prediction_value = (feature_vector.dot(parameter_vector))
		f.write(str(i))
		f.write(",")
		f.write(str(prediction_value))
		f.write("\n")
		i += 1
	f.close()

# reading the data
train_data = np.genfromtxt('data/train.csv', delimiter=',')
# removing the Metadata
train_data = train_data[1:]

# reading the test data
test_data = np.genfromtxt('data/test.csv', delimiter=',')
# removing the Metadata
test_data = test_data[1:]

# fixing up random values of parameter vector, learning rate and lambda
parameter_vector = np.random.rand(13,)
parameter_vector1 = np.random.rand(13,)
parameter_vector2 = np.random.rand(13,)
parameter_vector3 = np.random.rand(13,)

p = 2
lambda_value = 0.0001
learning_rate = 0.000006
num_of_iterations = 10000

p1 = 1.25
lambda_value1 = 0.0001
learning_rate1 = 0.000006
num_of_iterations1 = 10000

p2 = 1.5
lambda_value2 = 0.0001
learning_rate2 = 0.000006
num_of_iterations2 = 10000

p3 = 1.75
lambda_value3 = 0.0001
learning_rate3 = 0.000006
num_of_iterations3 = 10000

parameter_vector = get_new_regularised_parameter_vector(train_data, parameter_vector, learning_rate, lambda_value, p, num_of_iterations)
get_prediction(test_data, parameter_vector, "output.csv")

parameter_vector1 = get_new_regularised_parameter_vector(train_data, parameter_vector1, learning_rate1, lambda_value1, p1, num_of_iterations1)
get_prediction(test_data, parameter_vector1, "output_p1.csv")

parameter_vector2 = get_new_regularised_parameter_vector(train_data, parameter_vector2, learning_rate2, lambda_value2, p2, num_of_iterations2)
get_prediction(test_data, parameter_vector2, "output_p2.csv")

parameter_vector3 = get_new_regularised_parameter_vector(train_data, parameter_vector3, learning_rate3, lambda_value3, p3, num_of_iterations3)
get_prediction(test_data, parameter_vector3, "output_p3.csv")


# # new_predict_vector = get_prediction(test_data, parameter_vector)