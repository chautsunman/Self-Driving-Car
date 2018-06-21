import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.utils import Sequence
import pandas as pd
import cv2

EPOCHS = 4
BATCH_SIZE = 32

class DrivingLogSequence(Sequence):
  def __init__(self, log, batch_size):
    self.log = log
    self.batch_size = batch_size

  def __getitem__(self, idx):
    batch_start_index = self.batch_size * idx
    batch_end_index = self.batch_size * (idx + 1)

    batch = self.log.iloc[batch_start_index:batch_end_index]

    image_paths_batch = batch.iloc[:, 0:3].values
    steering_angle_batch = batch.iloc[:, 3].values

    # read image
    image_batch = []
    for i, image_paths in enumerate(image_paths_batch):
      image_type = np.random.randint(0, 3)
      image_batch.append(cv2.imread(image_paths[image_type]))
      if image_type == 1:
        steering_angle_batch[i] += 0.8
      elif image_type == 2:
        steering_angle_batch[i] -= 0.8
    # convert image color from BGR to YUV
    image_batch = [cv2.cvtColor(image, cv2.COLOR_BGR2YUV) for image in image_batch]
    for i, image in enumerate(image_batch):
      if np.random.rand() >= 0.8:
        # shift the image and steer more to teach the model to recover from off-center positions
        shift_magnitude = np.random.rand() * 2 - 1
        image_batch[i] = cv2.warpAffine(image, np.array([[1, 0, 160 * shift_magnitude], [0, 1, 0]], dtype="float64"), (image.shape[1], image.shape[0]))
        steering_angle_batch[i] += 0.4 * shift_magnitude
    for i, image in enumerate(image_batch):
      if np.random.rand() >= 0.5:
        # flip the image and the steering angle to teach the model to steer both left and right
        image_batch[i] = cv2.flip(image, 1)
        steering_angle_batch[i] *= -1

    # resize image
    image_batch = [cv2.resize(image, (200, 66)) for image in image_batch]

    return np.array(image_batch), steering_angle_batch

  def __len__(self):
    return int(np.floor(len(self.log) / self.batch_size))

def build_model():
  model = Sequential()
  model.add(Conv2D(
    input_shape=(66, 200, 3),
    filters=24,
    kernel_size=(5, 5),
    strides=(2, 2),
    activation="relu"
  ))
  model.add(Conv2D(
    filters=36,
    kernel_size=(5, 5),
    strides=(2, 2),
    activation="relu"
  ))
  model.add(Conv2D(
    filters=48,
    kernel_size=(5, 5),
    strides=(2, 2),
    activation="relu"
  ))
  model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation="relu"
  ))
  model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation="relu"
  ))
  model.add(Flatten())
  model.add(Dense(100, activation="relu"))
  model.add(Dense(50, activation="relu"))
  model.add(Dense(10, activation="relu"))
  model.add(Dense(1))

  return model

def train_model(model, log):
  model.compile(optimizer="adam", loss="mean_squared_error")

  model.fit_generator(
    DrivingLogSequence(log, BATCH_SIZE),
    epochs=EPOCHS,
    shuffle=True
  )

def main():
  # read the driving log
  log = pd.read_csv("driving_log.csv")

  # build the model
  model = build_model()

  # train the model
  train_model(model, log)

  # save the model
  model.save("model.h5")

if __name__ == "__main__":
  main()
