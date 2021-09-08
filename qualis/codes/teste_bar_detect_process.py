#Importacao das bibliotecas necessarias
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import time
# from picamera import PiCamera
# from picamera.color import Color


# Definicao da funcao morfologica
def morph_function(matinput):
  kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=2)
  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=1)

  return morph


# Analise das caracteristicas
def analyze_bars(matblobs,countours_frame, file):

  blobs,_ = cv2.findContours(matblobs,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
  valid_bars = []

  for i,bar in enumerate(blobs):
    rot_rect = cv2.minAreaRect(bar)
    b_rect = cv2.boundingRect(bar)

    (cx,cy),(sw,sh),angle = rot_rect
    rx,ry,rw,rh = b_rect

    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)

    on_count = cv2.contourArea(bar)
    total_count = sw*sh
    if total_count <= 0:
      continue

    if sh > sw :
      temp = sw
      sw = sh
      sh = temp

      

    # Area minima 
    if sw * sh < 400:
      continue

    # area maxima
    if sw * sh > 2000:
      continue  


    #print('Area: ', sw * sh)

    # Proporcao da barra
    rect_ratio = sw / sh

    # print('rect_ratio:', rect_ratio)

    if rect_ratio <= 3 or rect_ratio >= 15.5:
      continue

    # Desenhando o contorno da area da barra encontrada
    frame = cv2.drawContours(countours_frame,[box],0,(0,255,0),1)

    # Desenhando o contorno da area da barra encontrada
    #frame = cv2.drawContours(countours_frame,[box],0,(0,0,255),1)

    # Proporcao do preenchimento
    fill_ratio = on_count / total_count
    # print('fill_ratio: ', fill_ratio)

    # if fill_ratio < 0.6 :
    #   continue

    # print('countours_frame[int(cy),int(cx),0] ->', countours_frame[int(cy),int(cx),0])
    # # Remove as barras que sao mais claras
    # if countours_frame[int(cy),int(cx),0] > 100:
    #   continue

    valid_bars.append(bar)

  if valid_bars:
    print("O Arquivo {}, possui um total de {} linhas pretas".format(os.path.basename(file), len(valid_bars)))
  
  #print('-----fim --> ', format(os.path.basename(file)), '\n\n')
  # cv2.imshow("countours_frame_in",countours_frame)
  # cv2.waitKey(1)

  return valid_bars


def main_process():

  TIME_TAKE_PHOTO = time.strftime('%Y-%m-%d_%H:%M:%S') 

  # camera = PiCamera() 

  # camera.start_preview()
  # time.sleep(5)
  # camera.annotate_text = "Distancia-alvo: 5m\nTamanho-barras: 5cm\nDistancia-entre: 15cm"
  # camera.annotate_text_size = 30
  # camera.annotate_background = Color('black')
  # camera.capture('../imgs/in/third_tests/image_' + TIME_TAKE_PHOTO + '.jpg')
  # camera.stop_preview()

  for file in glob.glob("../imgs/in/third_tests/*"):
  # file = '../imgs/in/third_tests/image_' + TIME_TAKE_PHOTO + '.jpg'
    img = cv2.imread(file) 

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Funcao Gaussiana para adicionar ruido (blur)
    blurred = cv2.GaussianBlur(gray,(3,3),-1)
    
    # Aplicando a tecnica 'thresholding Otsu'
    # Extrair a caracteristica da binarizacao da imagem
    # Otsu thresholding     
    # ret, segment = cv2.threshold(blurred, 9, 100, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)  
    TESTS = {
      'T1'  : {
        'segment'  : cv2.adaptiveThreshold (blurred, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2),
        'descrition': ["AdaptiveThreshold()", "ADAPTIVE_THRESH_GAUSSIAN_C", "Type = THRESH_BINARY", "maxValue=190, blockSize=11, C=2"], 
      },
      'T2'  : {
        'segment'  : cv2.adaptiveThreshold (blurred, 195, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7,2),
        'descrition': ["AdaptiveThreshold()", "ADAPTIVE_THRESH_GAUSSIAN_C", "Type = THRESH_BINARY_INV", "maxValue=195, blockSize=11, C=2"], 
      },
      'T3'  : {
        'segment'  : cv2.adaptiveThreshold (blurred, 190, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5,2),
        'descrition': ["AdaptiveThreshold()", "ADAPTIVE_THRESH_GAUSSIAN_C", "Type = THRESH_TRUNC", "maxValue=190, blockSize=11, C=2"], 
      },
      'T4'  : {
        'segment'  : cv2.adaptiveThreshold (blurred, 185, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3,2),
        'descrition': ["AdaptiveThreshold()", "ADAPTIVE_THRESH_GAUSSIAN_C", "Type = THRESH_TOZERO", "maxValue=185, blockSize=11, C=2"], 
      },
      'T5'  : {
        'segment'  : cv2.adaptiveThreshold (blurred, 180, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13,2),
        'descrition': ["AdaptiveThreshold()", "ADAPTIVE_THRESH_GAUSSIAN_C", "Type = THRESH_TOZERO_INV", "maxValue=180, blockSize=11, C=2"], 
      },
      'T6'  : {
        'segment'  : cv2.adaptiveThreshold (blurred, 175, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15,5),
        'descrition': ["AdaptiveThreshold()", "ADAPTIVE_THRESH_GAUSSIAN_C", "Type = THRESH_MASK", "maxValue=175, blockSize=11, C=2"], 
      },
      'T7'  : {
        'segment'  : cv2.adaptiveThreshold (blurred, 190, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,2),
        'descrition': ["AdaptiveThreshold()", "ADAPTIVE_THRESH_GAUSSIAN_C", "Type = THRESH_OTSU", "maxValue=173, blockSize=11, C=2"], 
      },
      'T8'  : {
        'segment'  : cv2.adaptiveThreshold (blurred, 190, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11,2),
        'descrition': ["AdaptiveThreshold()", "ADAPTIVE_THRESH_GAUSSIAN_C", "Type = THRESH_TRIANGLE", "maxValue=170, blockSize=11, C=2"], 
      },
    }
    for key, method in TESTS.items():
      # print(key, method)
      segment = method['segment']

      cv2.rectangle(segment, (2,5), (455, 107), (0, 255, 0), -1)
      segment = cv2.putText(segment, method['descrition'][0], (5,27), cv2.FONT_HERSHEY_TRIPLEX, 0.7, 255)
      segment = cv2.putText(segment, method['descrition'][1], (5,55), cv2.FONT_HERSHEY_TRIPLEX, 0.7, 255)
      segment = cv2.putText(segment, method['descrition'][2], (5,77), cv2.FONT_HERSHEY_TRIPLEX, 0.7, 255)
      segment = cv2.putText(segment, method['descrition'][3], (5,99), cv2.FONT_HERSHEY_TRIPLEX, 0.7, 255)

      # cv2.imshow("thresholding",segment)
      # cv2.imwrite('../imgs/out/fourth_tests/segment_{}_{}'.format(key, os.path.basename(file)), segment)
      # cv2.waitKey(0)

      matmorph = morph_function(segment)
      # cv2.imshow("matmorph",matmorph)
      # cv2.waitKey(1)

      img_with_bars = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
      valid_bars = analyze_bars(matmorph,img_with_bars, file)


      for b in range(len(valid_bars)):
        cv2.drawContours(img_with_bars,valid_bars,b,(0,255,255),-1)

      # cv2.imshow("img_with_bars",img_with_bars)
      # cv2.imwrite('../imgs/out/fourth_tests/found_bars_{}'.format(os.path.basename(file)), img_with_bars)
      # cv2.waitKey(0)

      # fig1, plots = plt.subplots(1, 1, figsize=(35,40))
      # plots.set_title("Original Image")
      # plots.imshow(img, cmap = 'gray')

      # fig2, plots2 = plt.subplots(1, 1, figsize=(35,40))
      # plots2.set_title("Segmentation Image")
      # plots2.imshow(segment, cmap = 'gray')
      # # fig2.savefig('../imgs/in/third_tests/segmentation_' + TIME_TAKE_PHOTO + '.jpg')

      # fig3, plots3 = plt.subplots(1, 1, figsize=(35,40))
      # plots3.set_title("Bar finded")
      # plots3.imshow(img_with_bars, cmap = 'gray')
      # # fig3.savefig('../imgs/in/third_tests/found_bars_' + TIME_TAKE_PHOTO + '.jpg')

      #
      # plt.switch_backend('TkAgg') #TkAgg (instead Qt4Agg)

      plt.figure(figsize=(19, 9))
      plt.subplot(121),
      plt.imshow(segment,cmap = 'gray')
      plt.title('Segmentação'), plt.xticks([]), plt.yticks([])
      plt.subplot(122),
      plt.imshow(img_with_bars,cmap = 'gray')
      plt.title('Barras encontradas'), plt.xticks([]), plt.yticks([])
      
      # mng = plt.get_current_fig_manager()
      # mng.frame.Maximize(True)

      path_img = '../imgs/out/fourth_tests/{}_{}'.format(key, os.path.basename(file))
      plt.savefig(path_img)
    
      # plt.show()
    


if __name__ == '__main__':
  main_process()