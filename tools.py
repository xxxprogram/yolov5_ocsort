import cv2


def update_center_points(data, dic_center_points):
    '''
    更新坐标
    '''
    for row in data:
        x1, y1, x2, y2, cls_name, conf, obj_id = row[:7]

        # 计算中心点坐标
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 更新字典
        if obj_id in dic_center_points:
            # 判断列表长度是否超过30
            if len(dic_center_points[obj_id]) >= 30:
                dic_center_points[obj_id].pop(0)
            dic_center_points[obj_id].append((center_x, center_y))
        else:
            dic_center_points[obj_id] = [(center_x, center_y)]

    return dic_center_points




def res2OCres(results):
    lst_res = []
    if results is None:
        return lst_res
    for res in results.tolist():
        box = res[:4]
        conf = res[-2]
        cls = res[-1]
        lst_res.append([cls, conf, box])

    return list(lst_res)




def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_info(frame, tracker_bbox, dic_center_points, detect_track):
    for bbox in tracker_bbox:
        box = bbox[:4]
        id = int(bbox[6])
        cls_name = bbox[4]
        # if cls_id != '所有目标':
        #     cls_text, id_text = cls_id.split('_')
        #     if cls_text != cls_name and id_text != id:
        #         continue
        color = compute_color_for_labels(id)
        # 彩框 yolov5
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
        # track_id
        cv2.putText(frame, str(cls_name) + '_' + str(id), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, color, 2)

        # 画尾巴
        points = dic_center_points[id]
        # 跳过没有足够点的数据
        if len(points) < 2:
            continue
        if detect_track == '目标跟踪':
            # 根据数据点绘制直线
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                cv2.line(frame, p1, p2, color, 4)  # 在画布上绘制线段

    return frame



