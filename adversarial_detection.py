# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model
from scipy.special import expit, softmax
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

class AdversarialDetection:
    def __init__(self, model, attack_type, monochrome, classes):
        self.classes = len(classes)
        self.epsilon = 1
        self.graph = tf.compat.v1.get_default_graph()
        self.monochrome = monochrome

        if self.monochrome:
            self.noise = np.zeros((416, 416))
        else:
            self.noise = np.zeros((416, 416, 3))

        self.adv_patch_boxes = []
        self.fixed = False

        self.model = load_model(model)
        self.model.summary()
        self.attack_type = attack_type

        self.delta = None
        loss = None
        for out in self.model.output:
            # Targeted One Box
            if attack_type == "one_targeted":
                loss = K.max(K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 5]))

            # Targeted Multi boxes
            if attack_type == "multi_targeted":
                loss = K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 5])

            # Untargeted Multi boxes
            if attack_type == "multi_untargeted":
                # loss = tf.reduce_sum(K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 5:]))
                for i in range(0, self.classes):
                    if loss == None:
                        loss = tf.reduce_sum(K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, i+5]))
                    else:
                        loss = loss + tf.reduce_sum(K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, 4]) * K.sigmoid(K.reshape(out, (-1, 5 + self.classes))[:, i+5]))

            grads = K.gradients(loss, self.model.input)
            if self.delta == None:
                self.delta =  K.sign(grads[0])
            else:
                self.delta = self.delta + K.sign(grads[0])

        # Store current patches
        self.patches = []

        # loss = K.sum(K.abs((self.model.input-K.mean(self.model.input))))

        # Reduce Random Noises
        # loss = - 0.01 * tf.reduce_sum(tf.image.total_variation(self.model.input))

        # Mirror
        # loss = - 0.01 * tf.reduce_sum(tf.image.total_variation(self.model.input)) - 0.01 * tf.reduce_sum(K.abs(self.model.input - tf.image.flip_left_right(self.model.input)))

        grads = K.gradients(loss, self.model.input)
        self.delta = self.delta + K.sign(grads[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

    def attack(self, input_cv_image):
        with self.graph.as_default():
            # Draw each adversarial patch on the input image
            if not self.fixed:
                # Iterating the patch
                for box in self.adv_patch_boxes:
                    if self.monochrome:
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 0] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 1] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 2] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                    else:
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] = self.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :]
            else:
                # If the patch is fixed, just draw previous saved patches
                ib = 0
                for box in self.adv_patch_boxes:
                    if self.monochrome:
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 0] = self.patches[ib]
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 1] = self.patches[ib]
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 2] = self.patches[ib]
                    else:
                        input_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] = self.patches[ib]
                    ib = ib + 1

            if(len(self.adv_patch_boxes) > 0 and (not self.fixed)):
                grads = self.sess.run(self.delta, feed_dict={self.model.input:np.array([input_cv_image])}) / 255.0
                if self.monochrome:
                    # For monochrome images, we average the gradients over RGB channels
                    self.noise = self.noise + 5 / 3 * (grads[0, :, :, 0] + grads[0, :, :, 1] + grads[0, :, :, 2])
                else:
                    self.noise = self.noise + 5 * grads[0, :, :, :]
                # self.noise = np.clip(self.noise, 0.0, 1.0)

            return input_cv_image, self.sess.run(self.model.output, feed_dict={self.model.input:np.array([input_cv_image])})
