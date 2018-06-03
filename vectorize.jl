import TensorFlow

const tf = TensorFlow

 

rows = 2

columns = 5

@show rows, columns

 

# random matrix of size rows * columns

Xmatrix = tf.Variable(rand(rows, columns))

 

# get matrix dimensions

Xmatrix_shape = tf.get_shape(Xmatrix)

Xmatrix_rows = get(Xmatrix_shape.dims[1])

Xmatrix_columns = get(Xmatrix_shape.dims[2])

@show Xmatrix_rows, Xmatrix_columns

 

# reshape matrix = vectorize matrix

Xvector = reshape(Xmatrix, [Xmatrix_rows*Xmatrix_columns, 1])

 

# get dimensions of vectorized matrix

Xvector_shape = tf.get_shape(Xvector)

Xvector_rows = get(Xvector_shape.dims[1])

Xvector_columns = get(Xvector_shape.dims[2])

@show Xvector_rows, Xvector_columns

 

# reverse vectorization

Xmatrix_re = reshape(Xvector, [Xmatrix_rows, Xmatrix_columns])

Xmatrix_re = tf.get_shape(Xmatrix_re)

Xmatrix_re_rows = get(Xmatrix_re.dims[1])

Xmatrix_re_columns = get(Xmatrix_re.dims[2])

@show Xmatrix_re_rows, Xmatrix_re_columns

 

println()