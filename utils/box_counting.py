from ultralytics import YOLO
import numpy as np
import cv2

class BOXCOUNTING:
    def __init__(self, model_path: str, conf_threshold: float = 0.3, show: bool = False):
        """initialize

        Args:
            model_path (str): path to yolo model
            conf_threshold (float): confidence level for filtering. Defaults to 0.3.
            show (bool): use for debugging and show the model result. Defaults to False.
        """        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.show = show

        print(f"Model loaded from: {model_path}, using threshold: {conf_threshold}")


    def count(self, img: np.array) -> int:
        """counting a box from croppped image

        Args:
            img (np.array): input image of one pallet 

        Returns:
            int: number of the box in image
        """        
        results = self.model(img)

        count = 0
        boxes_by_class = {}

        front_class = 0
        
        for result in results:  # Iterate through the results
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract the bounding box coordinates
                conf = box.conf[0].item()  # Extract the confidence score
                cls = int(box.cls[0].item())  # Extract the class index

                if conf >= self.conf_threshold and cls == front_class:
                    if cls not in boxes_by_class:
                        boxes_by_class[cls] = []
                    boxes_by_class[cls].append((x1, y1, x2, y2, conf))
                    count += 1

                    if self.show:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'Class: {cls}, Conf: {conf:.2f}'
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if self.show:
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return count
