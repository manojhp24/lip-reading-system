"""
TensorFlow 2 / Keras Checkpoint Loader for Nick Nochnack's LipNet Model.
Restores weights from standard TensorFlow checkpoint format into the architecture.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, 
    Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten, Input
)

def build_lipnet_model() -> tf.keras.Model:
    """Builds the exact LipNet architecture from nicknochnack/LipNet."""
    inputs = Input(shape=(75, 46, 140, 1), name="mouth_input")
    
    # Layer 0: Conv3D 128
    x = Conv3D(128, 3, padding='same', activation='relu')(inputs)
    x = MaxPool3D((1, 2, 2))(x)
    
    # Layer 1: Conv3D 256
    x = Conv3D(256, 3, padding='same', activation='relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    # Layer 2: Conv3D 75
    x = Conv3D(75, 3, padding='same', activation='relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    # Temporal Reshape
    x = TimeDistributed(Flatten())(x)
    
    # Layer 3: Bidirectional LSTM 128
    x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    # Layer 4: Bidirectional LSTM 128
    x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    # Layer 5: Dense 41 (Vocabulary size + 1)
    outputs = Dense(41, kernel_initializer='he_normal', activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="NickNochnack_LipNet")
    return model

def load_trained_lipnet(checkpoint_prefix: str = "models/checkpoint") -> tf.keras.Model:
    """Instantiates the model and loads weights from the checkpoint."""
    model = build_lipnet_model()
    
    # Build variables by passing a dummy batch
    dummy = tf.zeros((1, 75, 46, 140, 1), dtype=tf.float32)
    _ = model(dummy)
    
    # Restore weights using tf.train.Checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    # Check for direct file or parent dir
    if os.path.exists(checkpoint_prefix + ".index"):
        ckpt.restore(checkpoint_prefix).expect_partial()
    elif os.path.exists(checkpoint_prefix):
        ckpt.restore(checkpoint_prefix).expect_partial()
        
    return model

