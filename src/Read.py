try:
    import cupy as np  # (activate this if GPU is used)
except ImportError:
    import numpy as np  # (activate this if CPU is used)

import torchvision
import torchvision.transforms as transforms

from mlxtend.data import loadlocal_mnist
import json


def Read_CIFAR10(par, Agent):
  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform)

  npixels = len(trainset.data[1][1])
  classes = trainset.classes

  par.num_features = npixels*npixels*3  # RGB!
  par.num_classes = len(classes)
  par.split_number = int(Agent)

  # Flatten the arrays
  x_train, x_test = np.array(trainset.data.reshape(
      [-1, par.num_features])), np.array(testset.data.reshape([-1, par.num_features]))
  y_train, y_test = np.array(trainset.targets), np.array(testset.targets)

  # Split data per agent
  x_train_agent = np.array(np.split(x_train, par.split_number))
  y_train_agent = np.array(np.split(y_train, par.split_number))

  return x_test, y_test, x_train, y_train, x_train_agent, y_train_agent


def Read_MNIST(par, Agent):
  ################################################################################################################################################
  ##### MNIST
  ##### type=np.ndarray (x_train: 60000 x 784 (float:0.~1.), y_train: 60000 x 1 (int:0~9), ... )
  ################################################################################################################################################
  par.num_features = 784  # 28*28
  par.num_classes = 10  # 0 to 9 digits
  par.split_number = int(Agent)
  # Load MNIST
  x_train, y_train = loadlocal_mnist(
      images_path='./Inputs/train-images-idx3-ubyte', labels_path='./Inputs/train-labels-idx1-ubyte')
  x_test, y_test = loadlocal_mnist(
      images_path='./Inputs/t10k-images-idx3-ubyte', labels_path='./Inputs/t10k-labels-idx1-ubyte')

  # Convert to float32.
  x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
  y_train, y_test = np.array(y_train), np.array(y_test)

  # Flatten images to 1-D vector of 784 features (28*28).
  x_train, x_test = x_train.reshape(
      [-1, par.num_features]), x_test.reshape([-1, par.num_features])

  # Normalize images value from [0, 255] to [0, 1].
  x_train, x_test = x_train / 255., x_test / 255.

  # Split data per agent
  x_train_agent = np.split(x_train, par.split_number)
  y_train_agent = np.split(y_train, par.split_number)
  x_train_agent, y_train_agent = np.array(
      x_train_agent), np.array(y_train_agent)

  x_list = []
  y_list = []
  for p in range(par.split_number):
    x_list.append(x_train_agent[p])
    y_list.append(y_train_agent[p])

  x_train_new = np.concatenate(np.array(x_list))
  y_train_new = np.concatenate(np.array(y_list))

  # FIXME: Isn't x_train equivalent to x_train? (and y_train too)
  return x_test, y_test, x_train_new, y_train_new, x_train_agent, y_train_agent


def Read_FEMNIST(par, Agent):
  ################################################################################################################################################
  ##### FEMNIST (datatype=dict)
  ##### Example: "users": ["f3795_00", "f3608_13"], "num_samples": [149, 162], "user_data": {"f3795_00": {"x": [], ..., []}, "y": [4, ..., 31]},
  ################################################################################################################################################
  par.num_features = 784  # 28*28
  par.num_classes = 62  # 0 to 9 digits + alphabet (26 + 26)
  TS = 36  # TS <= 36 for FEMNIST dataset

  train_data = {}
  x_train_agent = {}
  y_train_agent = {}
  tmp_x_train = []
  tmp_y_train = []
  tmpcnt = 0
  for testset in range(TS):
    with open('./Inputs/Femnist_Train_%s/all_data_%s_niid_05_keep_0_train_9.json' % (Agent, testset)) as f:
      train_data[testset] = json.load(f)
    for user in train_data[testset]["users"]:
      Temp_x_train = []
      Temp_y_train = []
      ## x_train
      for x_elem in train_data[testset]["user_data"][user]["x"]:
        tmp_x_train.append(1.0-np.array(x_elem))
        Temp_x_train.append(1.0-np.array(x_elem))
      Temp_x_train = np.array(Temp_x_train, np.float32)
      x_train_agent[tmpcnt] = Temp_x_train
      ## y_train
      for y_elem in train_data[testset]["user_data"][user]["y"]:
        tmp_y_train.append(np.array(y_elem))
        Temp_y_train.append(np.array(y_elem))
      Temp_y_train = np.array(Temp_y_train, np.uint8)
      y_train_agent[tmpcnt] = Temp_y_train
      tmpcnt += 1

  x_train_new = np.array(tmp_x_train, np.float32)
  y_train_new = np.array(tmp_y_train, np.uint8)

  x_train_new = np.array(tmp_x_train, np.float32)
  y_train_new = np.array(tmp_y_train, np.uint8)

  par.split_number = tmpcnt

  ## Testing
  test_data = {}
  temp_x_test = []
  temp_y_test = []
  total_num_test_data = 0
  for testset in range(TS):
    with open('./Inputs/Femnist_Test_%s/all_data_%s_niid_05_keep_0_test_9.json' % (Agent, testset)) as f:
      test_data[testset] = json.load(f)
    for user in test_data[testset]["users"]:
      total_num_test_data += len(test_data[testset]["user_data"][user]["y"])

      ## x_test
      for x_elem in test_data[testset]["user_data"][user]["x"]:
        temp_x_test.append(1.0-np.array(x_elem))

      ## y_test
      for y_elem in test_data[testset]["user_data"][user]["y"]:
        temp_y_test.append(y_elem)

  x_test = np.array(temp_x_test, np.float32)
  y_test = np.array(temp_y_test, np.uint8)

  return x_test, y_test, x_train_new, y_train_new, x_train_agent, y_train_agent
