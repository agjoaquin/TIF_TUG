# SitToStandApp.py
# ESTE SCRIPT DETECTA LA FLEXION DE RODILLA

import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from tkinter import Tk     
from tkinter.filedialog import askdirectory, askopenfilename
from pathlib import Path
from Calculator import filter_u_EMA, derivate_stack
import matplotlib.pyplot as plt

def processVideo():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # Abro una ventana para  seleccionar el archivo
    video_file = askopenfilename(title="Seleccione video a procesar") # show an "Open" dialog box and return the path to the selected file

    #Procesamiento de la dir del archivo para obtener datos como 
    #extensión de archivo, nombre de archivo y carpeta
    V_video_file = video_file.split("/")
    video_path = video_file[0:len(video_file)-len(V_video_file[-1])]
    V_video_file_name = V_video_file[-1].split(".")
    video_file_name = V_video_file[-1][0:len(V_video_file[-1])-len(V_video_file_name[-1])-1]
    video_file_extension = V_video_file_name[-1]

    # Carga de archivo
    #print( "Se analizará: " + video_file + " ...")
    cap = cv2.VideoCapture(video_file)
    print("VideoFile:"+video_file)
    # Datos del video cargado
    FPS_original = cap.get(cv2.CAP_PROP_FPS)   #ej. 25.0 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/FPS_original
    delta_t = 1/FPS_original
    width_original  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float "width"
    height_original = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float "height"
    resolution_original =  (int(width_original), int(height_original))  #ej. (640, 480)
    #fourcc_original = cap.get(cv2.CAP_PROP_FOURCC) #No se utilizar esta funcion
    #print("\n-FPS_original:" + str(FPS_original) + 
    #    "\n-frame_count_original: " + str(frame_count) + 
    #    "\n-duration_original:" + str(duration) + "s" 
    #    + "\n-resolution_original: " + str(int(width_original)) + "x" + str(int(height_original)) + "\n")

    # cv2.CAP_PROP_FRAME_WIDTH   # 3
    # cv2.CAP_PROP_FRAME_HEIGHT  # 4
    # cv2.CAP_PROP_FPS           # 5
    # cv2.CAP_PROP_FRAME_COUNT   # 7

    # Nombre Y ruta del video generado para guardar como RESULTADO
    video_path = str(Path.cwd())
    video_path_result = video_path + "/Videos/Videos Resultados/"
    video_file_result = video_path_result + video_file_name + "_resultado.mp4"

    # Datos para el video generado para guardar
    scale_percent = 700 * 100 / height_original # Porcentaje de escalado para el video a guardar (será el mismo para el video original a mostrar)
    FPS_result = FPS_original   #ej. 25.0
    width_result = int(width_original * scale_percent / 100)
    height_result = int(height_original * scale_percent / 100)
    resolution_result = (width_result, height_result)   #ej. (640, 480)
    frame_count_result = 0
    duration_result = 0

    # Creacion de los objetos para el guardado del video prosesado
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #*'mpv4"
    outVideoWriter = cv2.VideoWriter(video_file_result, fourcc, FPS_result, resolution_result) # (name.mp4, fourcc, FPS, resolution)

    # Vector de angulos de la rodilla (knee)
    V_angles_knee = np.zeros(0)
    V_angles_knee2 = np.zeros(0)
    V_vel_angles_knee = np.zeros(0)
    #V_angles_knee_filter = np.zeros(0)
    #V_vel_angles_knee_filter = np.zeros(0)
    alfa = 0.1
    # Vector de tiempos para cada frame
    V_time = np.zeros(0)

    # Inicio de While True para reproduccion y analisis
    with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose:
        while True:
            ret, frame = cap.read()

            if ret == False:
                break      
            
            V_time= np.append(V_time, frame_count_result/FPS_result)
            frame_count_result = frame_count_result + 1

            # Reescalado de la imagen/imagenes del video
            width = int(frame.shape[1] * scale_percent / 100)   # Otra opcion: height, width, layers = frame.shape
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)

            resized_frame = cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)   
            # resized_frame será el nuevo "frame" que se trabaja

            #Utilizo el primer frame como pantalla de carga
            if (frame_count_result == 1):
                    loading_page = resized_frame

            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            results = pose.process (frame_rgb)

            # Adquiero coordenadas de los marcadores
            if results.pose_landmarks is not None:
                # Landmark 24
                x1 = int(results.pose_landmarks.landmark[24].x * width)
                y1 = int(results.pose_landmarks.landmark[24].y * height)

                # Landmark 26
                x2 = int(results.pose_landmarks.landmark[26].x * width)
                y2 = int(results.pose_landmarks.landmark[26].y * height)

                # Landmark 28
                x3 = int(results.pose_landmarks.landmark[28].x * width)
                y3 = int(results.pose_landmarks.landmark[28].y * height)

                x4 = int(results.pose_landmarks.landmark[30].x * width)
                y4 = int(results.pose_landmarks.landmark[30].y * height)

                x5 = int(results.pose_landmarks.landmark[32].x * width)
                y5 = int(results.pose_landmarks.landmark[32].y * height)

                x6 = int(results.pose_landmarks.landmark[12].x * width)
                y6 = int(results.pose_landmarks.landmark[12].y * height)

                x11 = int(results.pose_landmarks.landmark[23].x * width)
                y11 = int(results.pose_landmarks.landmark[23].y * height)

                # Landmark 26
                x22 = int(results.pose_landmarks.landmark[25].x * width)
                y22 = int(results.pose_landmarks.landmark[25].y * height)

                # Landmark 28
                x33 = int(results.pose_landmarks.landmark[27].x * width)
                y33 = int(results.pose_landmarks.landmark[27].y * height)

                x44 = int(results.pose_landmarks.landmark[29].x * width)
                y44 = int(results.pose_landmarks.landmark[29].y * height)

                x55 = int(results.pose_landmarks.landmark[31].x * width)
                y55 = int(results.pose_landmarks.landmark[31].y * height)

                x66 = int(results.pose_landmarks.landmark[11].x * width)
                y66 = int(results.pose_landmarks.landmark[11].y * height)

                # Calculo de ángulo:
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])

                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # Calculo de ángulo:
                p11 = np.array([x11, y11])
                p22 = np.array([x22, y22])
                p33 = np.array([x33, y33])

                l11 = np.linalg.norm(p22 - p33)
                l22 = np.linalg.norm(p11 - p33)
                l33 = np.linalg.norm(p11 - p22)

                # Calcular el ángulo (teorema del coseno) y lo agrego a V_angles_knee
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                V_angles_knee = np.append(V_angles_knee, angle)

                # Calcular el ángulo (teorema del coseno) y lo agrego a V_angles_knee
                angle2 = degrees(acos((l11**2 + l33**2 - l22**2) / (2 * l11 * l33)))
                V_angles_knee2 = np.append(V_angles_knee2, angle2)

                # Visualización de segmentos de muslo y pierna
                aux_image = np.zeros(resized_frame.shape, np.uint8)

                #Rigth leg
                cv2.line(aux_image, (x1, y1), (x2, y2), (3, 202, 251), 20)
                cv2.line(aux_image, (x2, y2), (x3, y3), (3, 202, 251), 20)
                cv2.line(aux_image, (x3, y3), (x4, y4), (3, 202, 251), 20)
                cv2.line(aux_image, (x4, y4), (x5, y5), (3, 202, 251), 20)
                #Left leg
                cv2.line(aux_image, (x11, y11), (x22, y22), (3, 202, 251), 20)
                cv2.line(aux_image, (x22, y22), (x33, y33), (3, 202, 251), 20)
                cv2.line(aux_image, (x33, y33), (x44, y44), (3, 202, 251), 20)
                cv2.line(aux_image, (x44, y44), (x55, y55), (3, 202, 251), 20)
                #Hip
                #cv2.line(aux_image, (x1, y1), (x11, y11), (3, 202, 251), 20)
                #Shoulders
                #cv2.line(aux_image, (x6, y6), (x66, y66), (3, 202, 251), 20)
                #Torax
                #cv2.line(aux_image, (x1, y1), (x6, y6), (3, 202, 251), 20)
                #cv2.line(aux_image, (x11, y11), (x66, y66), (3, 202, 251), 20)


                contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
                
                #Output es el frame ya procesado
                output = cv2.addWeighted(resized_frame, 1, aux_image, 0.8, 0)   

                #Grafico landmarks con circulos
                cv2.circle(output, (x1, y1), 6, (5,5,170), 4)
                cv2.circle(output, (x2, y2), 6, (5,5,170), 4)
                cv2.circle(output, (x3, y3), 6, (5,5,170), 4)

                cv2.circle(output, (x11, y11), 6, (5,5,170), 4)
                cv2.circle(output, (x22, y22), 6, (5,5,170), 4)
                cv2.circle(output, (x33, y33), 6, (5,5,170), 4)

                # Agrego el angulo en el video
                cv2.putText(output, str(int(angle)), (x2, y2 - 30), 1, 1.5, (5,5,170), 2)   
                # Agrego info en el video
                cv2.putText(output, "Angulo en grados,", (10, height - 40), 4, 0.75, (20, 20, 20), 2) 

                #cv2.putText(output, "Pulse ESPACIO para finalizar.", (10, height - 10), 4, 0.75, (20, 20, 20), 2) 


                # Guardado del frame del video resultante
                outVideoWriter.write(output)

                #Pantalla de carga
                cv2.putText(loading_page, "Cargando: ", (10, height - 40), 4, 0.75, (20, 20, 20), 2) 
                #Barra de carga
                cv2.putText(loading_page, "|", (int((frame_count_result/frame_count)*(width-10)), height - 10), 4, 0.75, (0, 100, 0), 2) 
                # Muestro pantalla de carga  
                cv2.imshow("Cargando...", loading_page) 
                #Condicional para "Pulse espacio para terminar"
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break
            else:
                print("Skipped frame"+str(frame_count_result))


    # Guardo los angulos medidos
    #Filtro los angulos
    V_angles_knee_filter = filter_u_EMA(V_angles_knee,alfa) 
    V_angles_knee_filter2 = filter_u_EMA(V_angles_knee2,alfa) 
    #Calculo la velocidad con el ang filtrado
    V_vel_angles_knee = derivate_stack(V_angles_knee_filter,delta_t) 
    #Filtro la velocidad
    kernel_size = 20
    kernel = np.ones(kernel_size) / kernel_size
    V_vel_angles_knee_filter = np.convolve(V_vel_angles_knee, kernel, mode='same')

    delta_t
    print("FPS_original: "+str(FPS_original))
    print("delta_t: "+str(delta_t))
    print("V_angles_knee: "+str(V_angles_knee.size))
    print("V_time: "+str(V_time.size))
    print("V_angles_knee_filter : "+str(V_angles_knee_filter.size))
    print("V_vel_angles_knee_filter : "+str(V_vel_angles_knee_filter.size))

    V_ang_and_vel = np.stack(( V_time[:-2],V_angles_knee_filter[:-2], V_vel_angles_knee_filter[:-2]),1)
    data_path = video_path+"/Datos/"
    with open(data_path+'datos_ang_' + video_file_name + '.csv', 'wb') as h:
        np.savetxt(h, V_ang_and_vel, delimiter=',', fmt='%0.3f', header="Time (s),Ang (°),Vel (°/s)")


    plt.plot(V_time, V_angles_knee_filter)
    plt.plot(V_time, V_angles_knee_filter2)
    plt.xlabel("Tiempo")
    plt.ylabel("Ángulo")
    plt.title("Ángulos de ambas rodillas.")
    plt.legend(['Rodilla derecha','Rodilla Izquierda'])
    plt.show()

    #Calculo de la potencia muscular

    #Calculo de el tiempo promedio sit to stand
    # Rangos: //Varian según perspectiva de grabado
    #   Angular inferior : 70 _ 110 
    #   Angular superior : 135 _ 185
    #   Velocidad: -0,5 _ 0,5

    #Indicador de punto inferior, bool que indica si el paciente está sentado
    p_inf = 0 
    #Indicador de punto superior, bool que indica si el paciente está parado
    p_sup = 0 
    #Tiempo en punto inferior, donde el paciente dejo de estar sentado
    t_inf = 0 
    #Tiempo en punto superior, donde el paciente llegó a estar parado
    t_sup = 0 
    #Vector con diferenciales de tiempo
    V_t_dif = np.zeros(0) 
    #Contador para iterar vectores
    count = 0 
    while(np.size(V_time)>count):
        #Hallo punto inferior
        if (V_angles_knee_filter[count] > 70 and V_angles_knee_filter[count] < 110 and 
        V_vel_angles_knee_filter[count] > -0.5 and V_vel_angles_knee_filter[count] < 0.5 and
        p_sup == 0):
            #print("Halle un punto inferiror") 
            p_inf = 1
            t_inf = V_time[count]
        #Hallo punto superior #Mejorar
        if (V_angles_knee_filter[count] > 135 and V_angles_knee_filter[count] < 185 and 
        V_vel_angles_knee_filter[count] > -0.5 and V_vel_angles_knee_filter[count] < 0.5 and
        p_inf == 1):
            p_sup = 1
            t_sup = V_time[count]
        #Guardo el tiempo y lo agrego al vector
        if (p_inf == 1 and p_sup == 1):
            print("Tdif "+str(count)+": "+str(t_sup-t_inf))
            V_t_dif = np.append(V_t_dif,t_sup-t_inf)
            p_inf = 0
            p_sup = 0

        count=count+1  

    # Tiempo promedio que se tarda en ir desde 90 grados a 180 grados (levantarse)
    t_dif_prom = np.mean(V_t_dif)
    print("El tiempo promedio es: "+str(t_dif_prom))

    #Largo del femur [m] Ahora estimación, luego calculada o ingresada como input
    femur_lenght = 0.45

    #Potencia media
    Pmean = 2.733 - 6.228 * t_dif_prom + 18.224 * femur_lenght

    # Datos del video resultado generado
    # FPS, resolution y factor de escala ya se determinaron/seteados antes
    duration_result = frame_count_result/FPS_result
    print("\n-FPS_result: " + str(FPS_result) + 
        "\n-width_result: " + str(width_result) + 
        "\n-height_result: " + str(height_result) + 
        "\n-Resize Escala: " + str(scale_percent) +
        "\n-frame_count_result: " + str(frame_count_result) + 
        "\n-duration_result: " + str(duration_result))

    print("La potencia media es:"+str(Pmean))

    # RELEASE
    outVideoWriter.release()

    # Muestro imagenes/video
    video_out = cv2.VideoCapture(video_file_result)
    while(video_out.isOpened()):
        ret, frame = video_out.read()
        if ret==True:

            cv2.putText(frame, "La p.m.m. final es: "+str(Pmean), (10, height - 10), 4, 0.75, (20, 20, 20), 2) 
            cv2.imshow('Video procesado',frame)

            if cv2.waitKey(int(delta_t*1000)) & 0xFF == ord(' '):
                break
        else:
            break

    # RELEASE
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
    return 0