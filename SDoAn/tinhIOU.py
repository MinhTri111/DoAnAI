def get_iou(boxA, boxB, epsilon=1e-5):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
   
    # Tính diện tích phần giao nhau
    interArea = (xB - xA + 1) * (yB - yA + 1)
    
    # Diện tích của predicted và ground-truth bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Tính IoU = diện tích phần giao nhau chia diện tích phần tổng hợp
    # Diện tích phần hợp = tổng diện tích trừ diện tích phần giao
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


a=[1094.6        , 788.9                  ,1790.5              ,1267.1         ]
b=[1098.0         ,790.4                     ,1782     ,1287.4              ]
kq=get_iou(a,b)
print("%f" %kq)