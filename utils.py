from math import pi
import cv2

__all__ = ["load_image", "get_rad", "show_image", "save_image", "crop_roi"]


def load_image(img_path, shape=None):
    img = cv2.imread(img_path)
    if shape is not None:
        img = cv2.resize(img, shape)
    return img


def save_image(img_path, img):
    cv2.imwrite(img_path, img)


def show_image(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 720, 720)
    cv2.imshow(name, image)
    cv2.waitKey(0)


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))


def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))


def deg_to_rad(deg):
    return deg * pi / 180.0


def rad_to_deg(rad):
    return rad * 180.0 / pi


def crop_roi(img, gray=False):
    if gray:
        grey = img
    else:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    out = cv2.findContours(thresh, 1, 2)
    cnt = out[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y+h, x:x+w]
    return crop
