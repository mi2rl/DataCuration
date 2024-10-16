import tensorflow as tf

def DenseNet169(input_size=(512, 512, 3), classes=4):
    backbone = tf.keras.applications.densenet.DenseNet169(weights="imagenet",pooling='max', include_top=False, input_tensor=tf.keras.layers.Input(shape=input_size))
    backbone.outputs = [backbone.layers[-2].output]
    backbone_lastlayer = backbone.outputs[0]
    GAP_layer = tf.keras.layers.GlobalAveragePooling2D()(backbone_lastlayer)
    BN_layer = tf.keras.layers.BatchNormalization()(GAP_layer)
    dense_layer = tf.keras.layers.Dense(classes, activation='softmax')(BN_layer)
    model = tf.keras.models.Model(backbone.input, dense_layer)

    return model

def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), \
                       loss='sparse_categorical_crossentropy', \
                       metrics=['acc'])
#         print(model.summary())