import tensorflow as tf

# Create a TensorFlow constant
hello = tf.constant('Hello, TensorFlow!')

# No need to start a session or use sess.run() in TensorFlow 2.x
print(hello.numpy().decode())
