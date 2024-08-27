import tensorflow as tf 

#Tensor types of variables

# Constant
a = tf.constant(2)
b = tf.constant(3)

# Variables
x = tf.Variable(0, name='x')
y = tf.Variable(0, name='y')

string = tf.Variable("This is a string",tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

#Rank/Degree of a tensor
rank1_tensor = tf.Variable(["Test"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

#Shape of a tensor
tensor1 = tf.Variable([1,2,3], tf.int16)
tensor2 = tf.Variable([[1,2,3], [4,5,6]], tf.int16)

print(tensor1.shape)
print(tensor2.shape)

# Creating a 2D tensor
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32) 
print(tf.rank(tensor))
print(tensor.shape)

# Now lets select some different rows and columns from our tensor

three = tensor[0,2]  # selects the 3rd element from the 1st row
print(three)  # -> 3

row1 = tensor[0]  # selects the first row
print(row1)

column1 = tensor[:, 0]  # selects the first column
print(column1)

row_2_and_4 = tensor[1::2]  # selects second and fourth row
print(row_2_and_4)

column_1_in_row_2_and_3 = tensor[1:3, 0]
print(column_1_in_row_2_and_3)

