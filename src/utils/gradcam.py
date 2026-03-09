import torch
import cv2
import numpy as np


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)


    def save_activation(self, module, input, output):
        self.activations = output


    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]


    def generate(self, input_tensor):

        output = self.model(input_tensor)

        class_idx = output.argmax()

        self.model.zero_grad()
        output[:, class_idx].backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (224, 224))

        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam