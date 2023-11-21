import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('coins_2.JPG')
cv2.imshow("coins", img)
cv2.waitKey(0)

# Конвертуємо зображення в відтінки сірого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Порогова обробка
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Видаляємо шум
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Визначаємо фонову область
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Відстанційне перетворення
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Визначаємо невідому область
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Маркування міток
ret, markers = cv2.connectedComponents(sure_fg)

# Додаємо кожній маркованій області свій унікальний колір
colors = []
for i in range(1, ret + 1):
    colors.append(np.random.randint(0, 255, 3))

# Розфарбовуємо зображення відповідно до маркування
colored_markers = np.zeros_like(img)
for label in range(1, ret + 1):
    colored_markers[markers == label] = colors[label - 1]

# Додаємо невідому область червоним кольором
colored_markers[unknown == 255] = [0, 0, 255]

# Відображення результату
cv2.imshow("Coins Segmentation", colored_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Маркування міток
ret, markers = cv2.connectedComponents(sure_fg)
# Додайте один до всіх міток, щоб впевнений фон був не 0, а 1
markers = markers + 1
# Тепер позначте область невідомого нулем
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

cv2.imshow("coins_markers", img)
cv2.waitKey(0)