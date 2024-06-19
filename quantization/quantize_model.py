import tensorflow as tf

def quantize_model(model_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open('quantized_model.tflite', 'wb') as f:
        f.write(tflite_quant_model)

if __name__ == "__main__":
    quantize_model('example_model.h5')
