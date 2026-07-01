# IMPORT LIBRARIES
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image  #type: ignore 
import os


# LOADING OUR TRAINED MODEL

model_path = '04_models/SK_plant_model_v4_pro.keras'

if not os.path.exists(model_path):
    print(f" Error: Model file not found at {model_path}")
    exit()

model = tf.keras.models.load_model(model_path)

print("\nModel loaded successfully.\n")


# CLASS NAMES

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',

    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',

    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',

    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot'
]

def predict_my_plant(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return

    
    # LOAD AND PREPROCESS IMAGE :# 2. Image Preprocessing (AI ke size 150x150 aur scale mein convert karna)
    
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)        # Convert image to array
    img_array = img_array / 255.0                  # Normalize image 
    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension


    # 3. Prediction (AI ka guess)
    prediction = model.predict(img_array)

    # 4. Find the best guess
    score = prediction[0]    # Convert the prediction values to probabilities
    result_index = np.argmax(score)
    result_label = class_names[result_index]    # Get class name
    confidence = 100 * np.max(score)        # Get confidence score



    print(f"\n-----------------------------------")

    print(f"Testing Image: {os.path.basename(img_path)}")
    print(f"AI Guess: '{result_label}'")
    print(f"Confidence Score: {confidence:.2f}%")

    print(f"-----------------------------------\n")


# EXECUTION (Testing with a few photos)
# Path of photos :  
predict_my_plant("02_dataset/03_test/Apple healthy.jpg")         # New image :


# APPLE
predict_my_plant("02_dataset/03_test/1cfc6e73-1d86-4fb9-bffb-010163531711___FREC_C.Rust 3944_90deg.JPG") 
predict_my_plant(r"02_dataset\03_test\0ebf06b5-7471-42d7-b8d6-5a53127f90f3___RS_HL 8163_final_masked.jpg") 



# TOMATO
predict_my_plant("02_dataset/03_test/eca409c5-fd2b-4fab-9a54-4fad60a205d8___Crnl_L.Mold 8875.JPG") 
predict_my_plant(r"02_dataset\03_test\0e39c18a-9e04-4f6b-b557-dc0e944d22e5___RS_Erly.B 9501.JPG") 
predict_my_plant(r"02_dataset\03_test\0b36c5a2-6c9f-40d9-af4b-d4b0e66997da___Matt.S_CG 6653_final_masked.jpg") 


# POTATO
predict_my_plant(r"02_dataset\03_test\2d344be7-2d5f-4d27-aeb2-b92c8191b877___RS_Early.B 7173.JPG") 
predict_my_plant(r"02_dataset\03_test\0a0744dc-8486-4fbb-a44b-4d63e6db6197___RS_Early.B 7575.JPG") 
predict_my_plant(r"02_dataset\03_test\1c207156-339d-4ec9-9153-9edb3bc95b5f___RS_LB 4833.JPG") 
predict_my_plant(r"02_dataset\03_test\17a07992-ebb9-4368-8cb7-4af40d8ded08___RS_LB 3064.JPG") 



predict_my_plant(r"02_dataset\03_test\0e90fe4a-e8b6-4186-9429-a9fea180af9a___FREC_Scab 3391_270deg.JPG") 



# GRAPES
predict_my_plant(r"02_dataset\03_test\2b6ed613-e216-43ac-b518-ee54873b21e3___FAM_B.Rot 0639.JPG") # validation image 
predict_my_plant(r"02_dataset\03_test\grape healthy.jpg")
predict_my_plant(r"02_dataset\03_test\0f7500d4-cf47-4f9f-a505-455a4f7ca2e6___Mt.N.V_HL 9043.JPG")
predict_my_plant(r"02_dataset\03_test\1fd3808a-4e87-4cc0-8b5a-5e12de26c538___FAM_B.Rot 3253.JPG")




# CORN
predict_my_plant(r"02_dataset\03_test\1dbb4503-2025-4c04-98ac-eb8f49c51d12___R.S_HL 8221 copy.jpg") # validation image 
predict_my_plant(r"02_dataset\03_test\Corn_Common_Rust (94).JPG") # validation image 
predict_my_plant(r"02_dataset\03_test\HL@.jpg") # validation image 

predict_my_plant(r"02_dataset\03_test\0abbec2f-123f-4ae1-a9b7-8babbe8d0e89___RS_NLB 3685.JPG") # validation image 
predict_my_plant(r"02_dataset\03_test\2bfff79a-08ce-4f07-a059-465326ef2ea2___RS_GLSp 4436.JPG") # validation image 


