from ultralytics import YOLO
import numpy as np
import cv2

def iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2g - x1g + 1) * (y2g - y1g + 1)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

class BOXCOUNTING:
    def __init__(self, model_path: str, conf_threshold: float = 0.3, iou_threshold: float = 0.35, show: bool = False):
        """initialize

        Args:
            model_path (str): path to yolo model
            conf_threshold (float): confidence level for filtering. Defaults to 0.3.
            iou_threshold (float): IoU threshold for filtering overlapping boxes. Defaults to 0.35.
            show (bool): use for debugging and show the model result. Defaults to False.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.show = show

        print(f"Model loaded from: {model_path}, using confidence threshold: {conf_threshold} and IoU threshold: {iou_threshold}")

    def count(self, img: np.array) -> int:
        """counting a box from cropped image

        Args:
            img (np.array): input image of one pallet 

        Returns:
            int: number of the box in image
        """        
        top, bottom, left, right = [200] * 4

        # Add the black border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
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
