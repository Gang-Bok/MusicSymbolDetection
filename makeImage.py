import numpy as np
import os
import xml.etree.ElementTree as Et
import cv2
import pickle


def decode_mask(mask_string, shape):
    values = []
    for kv in mask_string.split(' '):
        k_string, v_string = kv.split(':')
        k, v = int(k_string), int(v_string)
        vs = [k for _ in range(v)]
        values.extend(vs)

    mask = np.array(values).reshape(shape)
    return mask


def make_image(xml_path, file_idx, margin=1):
    xml_path = xml_path
    xml = open(xml_path, "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    # file_name = root.attrib['document']
    objects = root.findall("Node")
    object_dict = {}
    for idx2, _object in enumerate(objects):
        Id = _object.find("Id").text
        object_dict[Id] = idx2

    chk = [False for _ in range(len(objects))]
    cnt = 1
    for idx2, _object in enumerate(objects):
        name = _object.find("ClassName").text
        crop_list = []
        bbox_list = []
        name_list = []
        if name != "staff":
            continue
        if chk[idx2]:
            continue
        chk[idx2] = True

        top = int(_object.find("Top").text)
        left = int(_object.find("Left").text)
        width = int(_object.find("Width").text)
        height = int(_object.find("Height").text)
        bottom = top + height
        right = left + width

        outlinks_str = _object.find("Outlinks").text
        outlinks = outlinks_str.split(' ')
        for _outlinks in outlinks:
            linked_object = objects[object_dict[_outlinks]]
            linked_name = linked_object.find("ClassName").text
            # print(linked_name)
            linked_top = int(linked_object.find("Top").text)
            linked_left = int(linked_object.find("Left").text)
            linked_width = int(linked_object.find("Width").text)
            linked_height = int(linked_object.find("Height").text)
            linked_bottom = linked_top + linked_height
            linked_right = linked_left + linked_width
            top = min(top, linked_top)
            left = min(left, linked_left)
            bottom = max(bottom, linked_bottom)
            right = max(right, linked_right)

        try:
            inlinks_str = _object.find("Inlinks").text
            inlinks = inlinks_str.split(' ')
        except:
            continue

        for _inlinks in inlinks:
            linked_object = objects[object_dict[_inlinks]]
            linked_name = linked_object.find("ClassName").text
            if linked_name == 'staffGrouping' or linked_name == 'measureSeparator':
                continue
            linked_top = int(linked_object.find("Top").text)
            linked_left = int(linked_object.find("Left").text)
            linked_width = int(linked_object.find("Width").text)
            linked_height = int(linked_object.find("Height").text)
            linked_bottom = linked_top + linked_height
            linked_right = linked_left + linked_width
            top = min(top, linked_top)
            left = min(left, linked_left)
            bottom = max(bottom, linked_bottom)
            right = max(right, linked_right)

            outlinks_str = linked_object.find("Outlinks").text
            outlinks = outlinks_str.split(' ')
            for _outlinks in outlinks:
                linked_object = objects[object_dict[_outlinks]]
                linked_name = linked_object.find("ClassName").text
                if linked_name[:5] == 'staff':
                    continue
                linked_top = int(linked_object.find("Top").text)
                linked_left = int(linked_object.find("Left").text)
                linked_width = int(linked_object.find("Width").text)
                linked_height = int(linked_object.find("Height").text)
                linked_bottom = linked_top + linked_height
                linked_right = linked_left + linked_width
                top = min(top, linked_top)
                left = min(left, linked_left)
                bottom = max(bottom, linked_bottom)
                right = max(right, linked_right)

        height = bottom - top + 2 * margin
        width = right - left + 2 * margin

        canvas = np.zeros((height, width), dtype='uint8')

        for _inlinks in inlinks:
            linked_object = objects[object_dict[_inlinks]]
            linked_name = linked_object.find("ClassName").text
            if linked_name == 'staffGrouping':
                continue
            if linked_name == 'barline':
                print(linked_name)
            linked_top = int(linked_object.find("Top").text)
            linked_left = int(linked_object.find("Left").text)
            linked_width = int(linked_object.find("Width").text)
            linked_height = int(linked_object.find("Height").text)
            _pt = linked_top - top + margin
            _pl = linked_left - left + margin
            linked_mask_str = linked_object.find("Mask").text

            mask = decode_mask(linked_mask_str, shape=(linked_height, linked_width))
            _pt = max(0, _pt)
            _pl = max(0, _pl)
            # _pl = min(_pl, width - linked_width)
            linked_width = min(linked_width, width - _pl)
            linked_height = min(linked_height, height - _pt)
            mask = mask[0:linked_height, 0:linked_width]
            mask = mask.astype('uint8')
            x_min, y_min, x_max, y_max = None, None, None, None
            try:
                if linked_name == 'measureSeparator':
                    crop_list.append(_pl)
                else:
                    canvas[_pt:_pt + linked_height, _pl:_pl + linked_width] += mask
                    if linked_name[:8] != 'notehead':
                        bbox_list.append([_pl, _pt, _pl + linked_width, _pt + linked_height])
                        name_list.append(linked_name)
                    else:
                        note_x_min = _pl
                        note_y_min = _pt
                        note_x_max = _pl + linked_width
                        note_y_max = _pt + linked_height
            except:
                print(linked_name)

            if linked_name[:8] == 'notehead':
                x_min = note_x_min
                y_min = note_y_min
                x_max = note_x_max
                y_max = note_y_max
                outlinks_str = linked_object.find("Outlinks").text
                outlinks = outlinks_str.split(' ')
                name_number = None
                hasDot = False
                if linked_name[:12] == 'noteheadFull':
                    name_number = 0
                else:
                    name_number = -1
                for _outlinks in outlinks:
                    linked_object = objects[object_dict[_outlinks]]
                    linked_name = linked_object.find("ClassName").text
                    if linked_name[-3:] == 'Dot':
                        hasDot = True
                    if linked_name[:5] == 'staff':
                        continue
                    if name_number >= 0:
                        if linked_name[:5] == "stem":
                            name_number = max(name_number, 1)
                        elif linked_name[:7] == "flag8th":
                            name_number = max(name_number, 2)
                        elif linked_name[:8] == "flag16th":
                            name_number = max(name_number, 3)
                        elif linked_name[:8] == "flag32nd":
                            name_number = max(name_number, 4)
                        elif linked_name[:8] == "flag64th":
                            name_number = max(name_number, 5)
                    else:
                        if linked_name[:5] == "stem":
                            name_number = min(name_number, -2)
                    linked_top = int(linked_object.find("Top").text)
                    linked_left = int(linked_object.find("Left").text)
                    linked_width = int(linked_object.find("Width").text)
                    linked_height = int(linked_object.find("Height").text)
                    _pt = linked_top - top + margin
                    _pl = linked_left - left + margin
                    linked_mask_str = linked_object.find("Mask").text

                    mask = decode_mask(linked_mask_str, shape=(linked_height, linked_width))
                    _pt = max(0, _pt)
                    _pl = max(0, _pl)
                    linked_width = min(linked_width, width - _pl)
                    linked_height = min(linked_height, height - _pt)
                    mask = mask[0:linked_height, 0:linked_width]
                    mask = mask.astype('uint8')
                    try:
                        canvas[_pt:_pt + linked_height, _pl:_pl + linked_width] += mask
                        if linked_name == 'stem':
                            if (note_y_min + note_y_max) < (_pt + _pt + linked_height):
                                x_min = min(x_min, _pl)
                                y_min = note_y_min
                                x_max = max(x_max, _pl + linked_width)
                                y_max = max(y_max, _pt + linked_height)
                            else:
                                x_min = min(x_min, _pl)
                                y_min = min(y_min, _pt)
                                x_max = max(x_max, _pl + linked_width)
                                y_max = note_y_max


                        elif linked_name[:4] == 'flag' or linked_name[-3:] == 'Dot':
                            x_min = min(x_min, _pl)
                            y_min = min(y_min, _pt)
                            x_max = max(x_max, _pl + linked_width)
                            y_max = max(y_max, _pt + linked_height)
                        else:
                            bbox_list.append([_pl, _pt, _pl + linked_width, _pt + linked_height])
                            name_list.append(linked_name)
                    except:
                        print(linked_name)
                name = None
                if name_number == 1:
                    name = "quarter_note"
                elif name_number == 2:
                    name = "8th_note"
                elif name_number == 3:
                    name = "16th_note"
                elif name_number == 4:
                    name = "32nd_note"
                elif name_number == 5:
                    name = "64th_note"
                elif name_number == -1:
                    name = "whole_note"
                elif name_number == -2:
                    name = "half_note"
                if hasDot:
                    if name is not None:
                        name = 'dot_' + name
                if name is not None:
                    bbox_list.append([x_min, y_min, x_max, y_max])
                    name_list.append(name)

        canvas[canvas > 0] = 1
        '''
        plt.imshow(canvas, cmap='gray', interpolation='nearest')
        plt.show()
        '''
        canvas_left = 0
        crop_list = sorted(list(set(crop_list)))
        canvas *= 255
        real_name_list = []
        file_name_list = []
        for crp_idx, crop_val in enumerate(crop_list):
            bbox = []
            crop_name_list = []
            canvas_right = crop_val
            try:
                crop_canvas = canvas[:, canvas_left:canvas_right]
                crop_canvas = cv2.cvtColor(crop_canvas, cv2.COLOR_GRAY2RGB)
                for _bbox, _name in zip(np.array(bbox_list), name_list):
                    if _bbox[0] >= canvas_left and _bbox[2] <= canvas_right:
                        cv2.rectangle(crop_canvas, (int(_bbox[0]) - canvas_left, int(_bbox[1])), (int(_bbox[2])-canvas_left, int(_bbox[3])),
                                      color=(0, 255, 0), thickness=1)
                        cv2.putText(crop_canvas, _name, (_bbox[0] - canvas_left, _bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        '''

                        '''
                        bbox.append([_bbox[0] - canvas_left, _bbox[1], _bbox[2] - canvas_left, _bbox[3]])
                        crop_name_list.append(_name)
                cv2.imshow('image', crop_canvas)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
                cv2.imwrite(r'images/' + xml_path[45:55] + str(cnt) + '_crop_' + str(crp_idx + 1) + '.png',
                            crop_canvas)
                np.save(r'bbox/bbox_' + xml_path[45:55] + str(cnt) + '_crop_' + str(crp_idx + 1), bbox)
                real_name_list.append(crop_name_list)
                file_name_list.append(xml_path[45:55] + str(cnt) + '_crop_' + str(crp_idx + 1))                
                '''

            except:
                print("bkp")
            canvas_left = canvas_right
        cnt += 1
        '''
        cv2.imshow('image', canvas * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return file_name_list, real_name_list


XML_PATH = r'muscima-pp/v2.1/data/annotations'
xml_file_list = os.listdir(XML_PATH)
img_dir_path_all = r'CVCMUSCIMA_WI/CVCMUSCIMA_WI/PNG_GT_BW'
img_file_list_all = os.listdir(img_dir_path_all)

if __name__ == "__main__":
    label_set = set()
    o_name_list = []
    f_name_list = []
    for idx, _xml_file in enumerate(xml_file_list):
        print(_xml_file)
        if _xml_file[-4:] == '.xsd':
            continue
        f_name, o_name = make_image(XML_PATH + '/' + _xml_file, idx)
        for _name_list, _f_name in zip(o_name, f_name):
            o_name_list.append(_name_list)
            f_name_list.append(_f_name)
            for _name in _name_list:
                label_set.add(_name)

    label_dic = {}
    for i, key in enumerate(label_set):
        label_dic[key] = i + 1
    for idx, _label in enumerate(o_name_list):
        label = []
        for _name in _label:
            label.append(label_dic[_name])
        np.save(r'label/label_' + f_name_list[idx], label)
    print(label_dic)
    with open('label_dic.pickle', 'wb') as fw:
        pickle.dump(label_dic, fw)
