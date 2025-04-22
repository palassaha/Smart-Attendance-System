try:
    import cmake
    print("try hard")
except ImportError as e:
    print(e)
    
import numpy
import cv2
print("NumPy version:", numpy.__version__)
print("OpenCV version:", cv2.__version__)
