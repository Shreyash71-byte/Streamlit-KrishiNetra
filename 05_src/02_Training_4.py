import os                                     
import tensorflow as tf
from tensorflow.keras import layers, models          # type: ignore 
from tensorflow.keras.layers import Dropout          # type: ignore 
from tensorflow.keras.preprocessing.image import ImageDataGenerator          # type: ignore 

# For Model 3 onwards 
from tensorflow.keras.callbacks import EarlyStopping                         # type: ignore 

# For Model 4
from tensorflow.keras.callbacks import ReduceLROnPlateau                     # type: ignore
from tensorflow.keras.layers import BatchNormalization                       # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint                       # type: ignore

# SECTION 1 — TRAIN DATA AUGMENTATION : “image transformation machine” ----->  image processing pipeline ban rahi hai.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,

    # Model 4 onwards
    brightness_range=[0.8,1.2],
    shear_range=0.1,
    fill_mode='nearest'
    )

val_datagen = ImageDataGenerator(rescale=1./255)

# SECTION 3: TRAIN GENERATOR
train_generator = train_datagen.flow_from_directory(
    '02_dataset/01_train',
    target_size=(180,180),
    batch_size=12,
    class_mode='categorical',
    shuffle=True
)

# SECTION 4: VALIDATION GENERATOR
val_generator = val_datagen.flow_from_directory(
    '02_dataset/02_validate',
    target_size=(180,180),
    batch_size=12,
    class_mode='categorical',
    shuffle=False
)


# 3. Build CNN Model 4 :

model = models.Sequential()

#-----------------------------------------------------------------------------------------------------------------------#

# C1: Convolution
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(180,180,3)))  #padding='same' ----> preserve edge information

model.add(BatchNormalization())

# C2: Convolution
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))  #padding='same' ----> preserve edge information

model.add(BatchNormalization())

# P1: Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2)))


#-----------------------------------------------------------------------------------------------------------------------#

# C3: Convolution
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))  #padding='same' ----> preserve edge information

model.add(BatchNormalization())

# C4: Convolution
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

# P2: Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2)))

#-----------------------------------------------------------------------------------------------------------------------#

# C5: Convolution
model.add(layers.Conv2D(filters= 128, kernel_size=(3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

# C6: Convolution
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

# P3: Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2)))

#-----------------------------------------------------------------------------------------------------------------------#

# Global Feature Aggregation
model.add(layers.GlobalAveragePooling2D())    # GlobalAveragePooling ----->reduce memorization

#-----------------------------------------------------------------------------------------------------------------------#

# Fully Connected Layers
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.3))                            # stronger dropout---------->better generalization
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.3))

#-----------------------------------------------------------------------------------------------------------------------#

# Output Layer
model.add(layers.Dense(19, activation='softmax'))


# 4. Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    '04_models/SK_plant_model_v4_pro.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# 5. TRAINING

print("\n Training started ........")
history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs=20,
    callbacks=[early_stop, lr_scheduler, checkpoint]
)
