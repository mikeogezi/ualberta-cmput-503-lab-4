import numpy as np

FILTERS = {
  'red': {
    'low': np.array([-10, 50, 50]),
    'high': np.array([10, 255, 255]),
  },
  'yellow': {
    'low': np.array([22, 93, 0]),
    'high': np.array([45, 255, 255]),
  },
#   'green': {
#     'low': np.array([50, 50, 50]),
#     'high': np.array([70, 255, 255]),
#   },
#   'blue': {
#     'low': np.array([110, 50, 50]),
#     'high': np.array([130, 255, 255]),
#   },
  'white': {
    'low': np.array([0, 0, 200]),
    'high': np.array([255, 55, 255]),
  }
}