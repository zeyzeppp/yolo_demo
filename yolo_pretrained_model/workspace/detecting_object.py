import cv2
import numpy as np

image = cv2.imread("resources/airplane.jpg")
#print(image.shape)
image = cv2.resize(image, (512, 760))
print(image.shape)

image_width = image.shape[1]
image_height = image.shape[0]

image_blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB = True, crop = False)

labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
            "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
            "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
            "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
            "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
            "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
            "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
            "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]


#colors = ["0, 255, 0", "255, 0, 0", "0, 0, 0", "255, 255, 0", "0, 0, 255", "255, 255, 0", "0, 0, 255"]
#colors = [np.array(color.split(",")).astype("int") for color in colors]
#colors = [np.array(colors)]
#colors = np.tile(colors, (16, 1))

model = cv2.dnn.readNetFromDarknet("pretrained_model/yolov3.cfg", "pretrained_model/yolov3.weights")

layers = model.getLayerNames()
output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

model.setInput(image_blob)

detection_layers = model.forward(output_layer)

for detection_layer in detection_layers:
    for object_detection in detection_layer:

        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]

        if(confidence > 0.9):
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
            (box_centerX, box_centerY, box_width, box_height) = bounding_box.astype("int")

            start_x = int(box_centerX - (box_width/2))
            start_y = int(box_centerY - (box_height/2))

            end_x = (start_x + box_width)
            end_y = (start_y + box_height)


            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255))
            cv2.putText(image, label, (start_x, start_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 3)


            #box_color = colors[predicted_id % len(colors)]
            #box_color = [int(each) for each in box_color]

            #cv2.rectangle(image, (start_x, start_y), (end_x, end_y), box_color)
            #cv2.putText(image, label, (start_x, start_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, box_color, 3)


cv2.imshow("demo", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
