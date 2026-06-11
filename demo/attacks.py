import os
# Set legacy Keras environment variable
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np

def fgsm_attack(model, input_image, label_idx, epsilon):
    """
    Generates adversarial example using Fast Gradient Sign Method (FGSM).
    """
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor([label_idx], dtype=tf.int64)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(label_tensor, prediction)
        
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    
    adv_x = input_image + epsilon * signed_grad
    adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
    
    return adv_x.numpy()

def pgd_attack(model, input_image, label_idx, epsilon, max_iter, alpha):
    """
    Generates adversarial example using Projected Gradient Descent (PGD).
    """
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    label_tensor = tf.convert_to_tensor([label_idx], dtype=tf.int64)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # 1. Random start within epsilon ball
    noise = tf.random.uniform(tf.shape(input_image), minval=-epsilon, maxval=epsilon)
    x_adv = input_image + noise
    x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    
    for _ in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            prediction = model(x_adv)
            loss = loss_object(label_tensor, prediction)
            
        gradient = tape.gradient(loss, x_adv)
        signed_grad = tf.sign(gradient)
        
        # Take step
        x_adv = x_adv + alpha * signed_grad
        # Project back to epsilon ball of input_image
        x_adv = tf.clip_by_value(x_adv, input_image - epsilon, input_image + epsilon)
        # Clip to valid range [0, 1]
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        
    return x_adv.numpy()

