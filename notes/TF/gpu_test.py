import tensorflow as tf
print(tf.__version__)
print(tf.test.gpu_device_name())

gpus = tf.config.experimental.list_physical_devices('GPU')
print(len(gpus))
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
