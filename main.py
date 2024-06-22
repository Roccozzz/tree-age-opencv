import cv2
import matplotlib.pyplot as plt
import math

def adjust_contrast_brightness(img, contrast=1, brightness=0):
    brightness += math.floor(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

def count_column(img, col):
    counter = 0
    isBlob = False

    for i in range(img.shape[0]):
        if img[i][col] > 0:
            isBlob = True
        if img[i][col] < 1:
            if isBlob:
                counter += 1
            isBlob = False

    return counter

image_path = 'img/input.tif'
image = cv2.imread(image_path)

plt.figure(figsize=(15, 10))
plt.tight_layout()

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=15)
mask = cv2.bitwise_not(closing)

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
plt.title("Mask")
plt.axis('off')

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    cont = max(contours, key=cv2.contourArea)
else:
    cont = []

image_contour = image.copy()
cv2.drawContours(image_contour, [cont], -1, (255, 0, 0), 7)
x, y, w, h = cv2.boundingRect(cont)
cv2.rectangle(image_contour, (x, y), (x+w, y+h), (0, 0, 255), 7)

# Plot image with trunk contour
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(image_contour, cv2.COLOR_BGR2RGB))
plt.title("Contour")
plt.axis('off')

crop = image[y:y+h, x:x+w]

crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

contrast = adjust_contrast_brightness(crop_gray, contrast=5, brightness=-30)
bw = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, -1)

plt.subplot(2, 3, 4)
plt.imshow(bw, cmap='gray')
plt.title("Enhanced Contrast")
plt.axis('off')

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
horizontal = cv2.erode(bw, horizontalStructure)

plt.subplot(2, 3, 5)
plt.imshow(horizontal, cmap='gray')
plt.title("Vertical Erosion")
plt.axis('off')

ringSamples = []
highestSample = 0
iHighestSample = 0

for i in range(15):
    col_index = math.floor(horizontal.shape[1] * ((i / 15) + 0.05))
    ring = count_column(horizontal, col_index)
    ringSamples.append(ring)

    if ring > highestSample:
        highestSample = ring
        iHighestSample = i

    plt.axvline(x=col_index, color="green")

col_index_highest = math.floor(horizontal.shape[1] * ((iHighestSample / 15) + 0.05))
plt.axvline(x=col_index_highest, color="red")

for i in range(10):
    print(f"Column {i}: {ringSamples[i]}")

centerRings = []
for i in range(-50, 50):
    col_index = math.floor(horizontal.shape[1] * ((iHighestSample / 15) + 0.05) + (i / 1000))
    age = count_column(horizontal, col_index)
    centerRings.append(age)

avg_rings = math.floor(sum(centerRings) / len(centerRings))
print(f"\nTree rings: {avg_rings}")

plt.suptitle(f"Tree rings: {avg_rings}", fontsize=16)

plt.subplot(2, 3, 6).axis('off')
plt.show()
