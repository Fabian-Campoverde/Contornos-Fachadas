import os, uuid
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
import cv2
import numpy as np
import torch
from PIL import Image
from transparent_background import Remover
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.swinir_arch import SwinIR
import math


class Dexined1:
    def __init__(self, modelPath='funciones/edge_detection_dexined_2024sep.onnx', backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId
        
        # Cargar el modelo
        self._model = cv2.dnn.readNetFromONNX(self._modelPath)
        self.setBackendAndTarget(self._backendId, self._targetId)

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def postProcessing(self, output, shape):
        h, w = shape
        preds = [self.sigmoid(p) for p in output]
        preds = [cv2.normalize(np.squeeze(p), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) for p in preds]
        preds = [cv2.resize(p, (w, h)) for p in preds]
        fuse = preds[-1]
        ave = np.uint8(np.mean(preds, axis=0))
        return fuse, ave

    def infer(self, image):
        inp = cv2.dnn.blobFromImage(image, 1.0, (512, 512), (103.5, 116.2, 123.6), swapRB=False, crop=False)
        self._model.setInput(inp)
        out = self._model.forward()
        result, _ = self.postProcessing(out, image.shape[:2])
        _, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)
        return result

def bordearDexi(imagen_cv):
    modelo = Dexined1()
    bordes = modelo.infer(imagen_cv)
    return bordes

def comprimir(imagen):
    # Cargar imagen con OpenCV
    if not isinstance(imagen, np.ndarray):
        img_pil = Image.open(imagen).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        img = imagen

    # Redimensionar si es muy grande
    MAX_WIDTH = 800
    height, width = img.shape[:2]

    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def mejorarGAN(imagen_cv):
    model_path = "weights/RealESRGAN_x4plus.pth"  # Asegúrate de tener el archivo

    # --------- CARGAR IMAGEN ---------
    img_np= comprimir(imagen_cv)
    

    # --------- CONFIGURAR MODELO ---------
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
        num_grow_ch=32, scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False 
    )

    # --------- MEJORAR IMAGEN ---------
    output, _ = upsampler.enhance(img_np, outscale=3.5)
    result=cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return result

def mejorarSwim(imagen):
    
    model_path = "model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN-with-dict-keys-params-and-params_ema.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='1conv'
    ).to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)
    model.eval()
    def mod_crop(img, scale):
        h, w = img.shape[-2:]
        h = h - h % scale
        w = w - w % scale
        return img[..., :h, :w]

    # ---------- CARGAR IMAGEN ----------
    img = comprimir(imagen)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalizar a [0,1]
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(device)
    img_tensor = mod_crop(img_tensor, 8)

    # ---------- PROCESAR ----------
    with torch.no_grad():
        output = model(img_tensor)
        output = output.clamp(0, 1)

    # ---------- GUARDAR IMAGEN ----------
    output_img = output.squeeze().cpu().numpy()
    output_img = np.transpose(output_img, (1, 2, 0)) * 255.0
    output_img = output_img.astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    return output_img

def medir(imagen):
    PIXEL_TO_METER = 0.02  # por ejemplo, 1 píxel = 2 cm

    
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Binarizar
    _, threshed = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

    # Escoger el contorno más grande
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # Aproximar contorno
    arclen = cv2.arcLength(cnt, True)
    proporcion=0.005
    epsilon = arclen * proporcion
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Dibujar líneas y mostrar medidas
    canvas = imagen.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(approx)):
        pt1 = tuple(approx[i][0])
        pt2 = tuple(approx[(i + 1) % len(approx)][0])  # siguiente punto (cerrar polígono)
        
        # Dibujar línea
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 2)

        # Calcular distancia en píxeles
        dist_px = math.dist(pt1, pt2)

        # Convertir a metros (si quieres)
        dist_m = dist_px * PIXEL_TO_METER

        # Calcular punto medio
        mid_x = int((pt1[0] + pt2[0]) / 2)
        mid_y = int((pt1[1] + pt2[1]) / 2)

        # Mostrar texto con medida
        text = f"{dist_m:.2f} m"
        cv2.putText(canvas, text, (mid_x, mid_y), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return canvas
    

def procesar_imagen(request):
    
    if request.method == 'POST':
        imagen = request.FILES.get('image')
        accion = request.POST.get('action')
        if not imagen:
            request.session['error'] = 'No se proporcionó ninguna imagen'
            return redirect('index')

        try:
            if accion == 'remover': 
                remover = Remover()
                img = Image.open(imagen)
                img_sin_fondo = remover.process(img)

                # Si tiene canal alfa, poner fondo negro
                if img_sin_fondo.mode == 'RGBA':
                    fondo_negro = Image.new('RGBA', img_sin_fondo.size, (0, 0, 0, 255))
                    fondo_negro.paste(img_sin_fondo, (0, 0), img_sin_fondo)
                    resultado = fondo_negro.convert('RGB')
                else:
                    resultado = img_sin_fondo
                filename = f"{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(settings.MEDIA_ROOT, 'results', filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                resultado.save(output_path)
               
            elif accion == 'bordear':
                img_pil = Image.open(imagen).convert('RGB')
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                resultado = bordearDexi(img_cv)
            
                filename = f"{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(settings.MEDIA_ROOT, 'results', filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, resultado)

            elif accion == 'mejorar_gan':
                
                resultado = mejorarGAN(imagen)
                filename = f"{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(settings.MEDIA_ROOT, 'results', filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, resultado)
            elif accion == 'mejorar_s':
                img_pil = Image.open(imagen).convert('RGB')
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                resultado = mejorarSwim(img_cv)
                filename = f"{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(settings.MEDIA_ROOT, 'results', filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, resultado)
            elif accion == 'medir':
                img_pil = Image.open(imagen).convert('RGB')
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                resultado = medir(img_cv)
                filename = f"{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(settings.MEDIA_ROOT, 'results', filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, resultado)

            # Guardar la URL en sesión
            request.session['image_url'] = f"/media/results/{filename}"

        except Exception as e:
            request.session['error'] = str(e)

        return redirect('index')  # Redirigir al mismo template (GET)

    # Método GET
    image_url = request.session.pop('image_url', None)
    error = request.session.pop('error', None)
    return render(request, 'index.html', {
        'image_url': image_url,
        'error': error
    })

