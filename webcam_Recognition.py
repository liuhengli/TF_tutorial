# encoding: utf-8


"""
@version: ??
@author: Liu Hengli
@license: Apache Licence 
@contact: liuhengli@gmail.com
@site: https://github.com/liuhengli
@software: PyCharm
@file: webcam_Recognition.py
@time: 2017/6/1 10:44
实现对摄像头/视频等人脸进行实时检测与识别
"""

import tensorflow as tf
import cv2
import align_dlib as align
import numpy as np
import facenet
import time

def webcam_recognizer(recognizer_model_file, face_detect_file, template_file):
    """
    实现实时人脸检测与识别
    :param recognizer_model_file: facenet模型路径
    :param face_detect_file: dlib人脸特征点检测模型路径
    :param template_file: 数据库模板路径
    :return:
        no retrun
    """
    #首先加载模型
    #加载dlib模型
    face_detect = align.AlignDlib(face_detect_file)
    landmarkIndices = align.AlignDlib.OUTER_EYES_AND_NOSE

    #加载facenet模型
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(recognizer_model_file)

            #获取输入输出参数以及输入数据的大小
            images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            image_size = int(images_placeholder.get_shape()[1])

            #初始化template_file参数
            template_face_num = 0
            template_face_label = []
            # template_face_emb =[]
            template_face_list = []
            #获取template_file的数据
            with open(template_file) as f:
                #读取模板数据
                template_face = f.readlines()
                template_face_num = len(template_face)
                for i in range(template_face_num):
                    face_and_label = template_face[i].strip('\n')
                    face_file, label = face_and_label.split(';')
                    template_face_label.append(int(label))
                    print("template_face_label", template_face_label)
                    face_image = cv2.imread(face_file)
                    print(face_file,"正在读取 ...")
                    template_aligned_face = face_detect.align(image_size,
                                                     face_image,
                                                     landmarkIndices=landmarkIndices,
                                                     skipMulti=False)
                    if template_aligned_face is None:
                        return False
                    #白化预处理
                    prewhitened = facenet.prewhiten(template_aligned_face)
                    template_face_list.append(prewhitened)
            f.close()
            print("template_face_list", np.shape(template_face_list))
            feed_dict = {images_placeholder: template_face_list, phase_train_placeholder: False}
            template_face_embeds = sess.run(embeddings, feed_dict=feed_dict)
            print("template_face_emb", np.shape(template_face_embeds))

            #开启摄像头
            video_capture = cv2.VideoCapture(1)
            video_capture.set(3, 600)
            video_capture.set(4, 480)

            print("开始人脸识别！！！")

            while video_capture.isOpened():
            # while True:
                # frame = cv2.imread('./zhonghanliang1.jpg')
                ret, frame = video_capture.read()
                if not ret:
                    break
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #获取每一帧图像的人脸
                # face_list = []

                aligned_faces = []
                # bbox_list = []
                start = time.time()
                bboxes = face_detect.getAllFaceBoundingBoxes(rgbImage)
                # print("bboxes", bboxes)
                if bboxes is None:
                    print("继续")
                    continue
                for box in range(len(bboxes)):
                    aligned_faces.append(face_detect.align(image_size,
                                                           rgbImage,
                                                           bboxes[box],
                                                           landmarkIndices=landmarkIndices))

                if aligned_faces == []:
                    continue
                # print("len(bboxes)", len(bboxes))
                print("人脸数：", len(aligned_faces))
                faces = []
                for num in range(len(aligned_faces)):
                    prewhitened_face = facenet.prewhiten(aligned_faces[num])
                    faces.append(prewhitened_face)
                feed_dict = {images_placeholder: faces,
                             phase_train_placeholder: False}
                face_embeddings = sess.run(embeddings, feed_dict=feed_dict)
                # print("face_embeddings", np.shape(face_embeddings))

                if not face_embeddings == []:
                    # print("----bboxes[0]", bboxes[num])
                    label = 999
                    # confidence_list = []
                    num_faces = len(face_embeddings)
                    # dist_arr = np.zeros((num_faces, template_face_num))
                    min_dist = []
                    labels = []
                    confidences = []
                    for k in range(num_faces):
                        # print("emb", face_embeddings[k])
                        dist_list = []
                        for j in range(template_face_num):
                            dist = np.sqrt(np.mean(np.square(np.subtract(face_embeddings[k],
                                                                         template_face_embeds[j, :]))))
                            # print("dist", dist)
                            dist_list.append(dist)
                        #计算最小距离
                        min_dist.append(np.min(dist_list))
                        min_dist_index = np.argmin(dist_list)
                        # print("min_dist_index", min_dist_index)
                        labels.append(template_face_label[min_dist_index])
                        # print(labels)
                        #计算置信度
                        # print("dist_list", dist_list)
                        # confidence_list = np.exp(-3 * np.array(dist_list))
                        # print("confidence_list", confidence_list)
                        # confidences.append(confidence_list[min_dist_index])
                    # print("min_dist", min_dist)
                    # print("labels", labels)
                    # print("confidences", confidences)
                    print("时间：%.3f" % (time.time()-start))
                    for l in range(len(labels)):
                        if labels[l] == 1:
                            pred = 'tangyang'
                        else:
                            pred = 'zhonghangliang'
                        # if confidences[l] > 0.72 and min_dist[l] < 0.1:
                        if  min_dist[l] < 0.085:
                            #计算人脸框的位置
                            left = bboxes[l].left()
                            top = bboxes[l].top()
                            bottom = bboxes[l].bottom()
                            right = bboxes[l].right()
                            text = pred + " " +str('%.3f'%min_dist[l])
                            cv2.putText(frame, text, (left, top-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)

                #显示
                cv2.imshow(" ", frame)
                cv2.waitKey(35)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            video_capture.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    recognizer_model_file = './20170512-110547/'
    dlib_face_predictor = './20170512-110547/shape_predictor_68_face_landmarks.dat'
    template_file = './templates/templates_describe_csv.txt'
    webcam_recognizer(recognizer_model_file, dlib_face_predictor, template_file)














