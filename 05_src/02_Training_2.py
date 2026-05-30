import os                                     
import tensorflow as tf
from tensorflow.keras import layers, models          # type: ignore 
from tensorflow.keras.layers import Dropout          # type: ignore 
from tensorflow.keras.preprocessing.image import ImageDataGenerator          # type: ignore 


# SECTION 1 — TRAIN DATA AUGMENTATION : “image transformation machine” ----->  image processing pipeline ban rahi hai.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    )

val_datagen = ImageDataGenerator(rescale=1./255)

# SECTION 3: TRAIN GENERATOR
train_generator = train_datagen.flow_from_directory(
    '02_dataset/01_train',
    target_size=(150,150),
    batch_size=8,
    class_mode='categorical',

)

# SECTION 4: VALIDATION GENERATOR
val_generator = val_datagen.flow_from_directory(
    '02_dataset/02_validate',
    target_size=(180,180),
    batch_size=10,
    class_mode='categorical',

)


# 3. Build CNN Model 2 :

model = models.Sequential()

#-----------------------------------------------------------------------------------------------------------------------#

# C1: Convolution
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(180,180,3)))  

# C2: Convolution
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))  


# P1: Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2)))


#-----------------------------------------------------------------------------------------------------------------------#

# C3: Convolution
model.add(layers.Conv2D(filters= 64, kernel_size=(3,3), activation='relu'))  


# C4: Convolution
model.add(layers.Conv2D(filters= 64, kernel_size=(3,3), activation='relu'))


# P2: Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2)))

#-----------------------------------------------------------------------------------------------------------------------#

# C5: Convolution
model.add(layers.Conv2D(filters= 64, kernel_size=(3,3), activation='relu'))


# C6: Convolution
model.add(layers.Conv2D(filters= 64, kernel_size=(3,3), activation='relu'))


# P3: Max Pooling
model.add(layers.MaxPooling2D(pool_size=(2,2)))

#-----------------------------------------------------------------------------------------------------------------------#

# Flatten
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



# 5. TRAINING

print("\n Training started ........")
history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs=20,
)


# 6. SAVE MODEL
if not os.path.exists('04_models'):
    os.makedirs('04_models')

model.save('04_models/plant_model_v2_pro.keras')
print("\n Pro Model save ho gaya: 04_models/plant_model_v2_pro.keras")

