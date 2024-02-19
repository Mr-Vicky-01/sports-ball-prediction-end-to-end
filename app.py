import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("Sports_ball_prediction_v2.h5")
labels = ['american_football', 'baseball', 'basketball', 'billiard_ball', 'bowling_ball', 'cricket_ball',
                     'football', 'golf_ball', 'hockey_ball', 'hockey_puck', 'rugby_ball', 'shuttlecock',
                     'table_tennis_ball', 'tennis_ball', 'volleyball']

def predict_image(inp):
    if inp is not None:
        img = tf.image.resize(inp, (224, 224))
    else:
        print("Input is None. Unable to resize.")
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)[0]
    
    return {labels[i]: float(prediction[i]) for i in range(len(labels))}

demo = gr.Interface(fn=predict_image, inputs='image',outputs=gr.Label(num_top_classes=3),title='Sports Ball Classification')
demo.launch(debug=True,share=True)