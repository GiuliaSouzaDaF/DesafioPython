from flask import Flask, render_template, Response, jsonify, request, redirect
import cv2
import mediapipe as mp
from datetime import datetime
import os

app = Flask(__name__)

# =========================
# VARIÁVEIS
# =========================

gesto_atual = "nenhum"
maquina_ligada = False
historico = []
usuario_atual = "Usuário"

# controla o último gesto detectado
ultimo_gesto = "nenhum"

# controla a última captura feita
ultimo_gesto_salvo = "nenhum"

mostrar_imagem = False

# =========================
# PASTA DE CAPTURAS
# =========================

PASTA_CAPTURAS = "capturas"

if not os.path.exists(PASTA_CAPTURAS):
    os.makedirs(PASTA_CAPTURAS)

# =========================
# MEDIAPIPE
# =========================

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils

maos = mp_maos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# DETECTORES
# =========================


def eh_joinha(p):
    polegar = (
        p[4].y < p[3].y and
        p[3].y < p[2].y
    )

    dedos = (
        p[8].y > p[6].y and
        p[12].y > p[10].y and
        p[16].y > p[14].y and
        p[20].y > p[18].y
    )

    return polegar and dedos


def eh_hangloose(p):
    polegar = abs(p[4].x - p[2].x) > 0.12

    mindinho = (
        p[20].y < p[18].y
    )

    dedos = (
        p[8].y > p[6].y and
        p[12].y > p[10].y and
        p[16].y > p[14].y
    )

    return polegar and mindinho and dedos


def eh_dedo_meio(p):
    meio = (
        p[12].y < p[10].y
    )

    indicador = (
        p[8].y > p[6].y
    )

    anelar = (
        p[16].y > p[14].y
    )

    mindinho = (
        p[20].y > p[18].y
    )

    return (
        meio and
        indicador and
        anelar and
        mindinho
    )


def salvar_captura(frame, gesto):
    hora = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    nome = f"{gesto}_{hora}.jpg"

    caminho = os.path.join(
        PASTA_CAPTURAS,
        nome
    )

    cv2.imwrite(caminho, frame)


@app.route("/", methods=["GET", "POST"])
def login():
    global usuario_atual

    if request.method == "POST":
        usuario_atual = request.form["nome"]
        return redirect("/maquina")

    return render_template("login.html")


@app.route("/maquina")
def maquina():
    return render_template(
        "index.html",
        usuario=usuario_atual
    )


def registrar(gesto):
    global maquina_ligada

    if gesto == "joinha":
        maquina_ligada = True

    elif gesto == "hangloose":
        maquina_ligada = False

    historico.append({
        "usuario": usuario_atual,
        "horario": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "gesto": gesto,
        "estado": "Ligada" if maquina_ligada else "Desligada"
    })


def gerar_frames():
    global gesto_atual
    global ultimo_gesto
    global ultimo_gesto_salvo
    global mostrar_imagem

    camera = cv2.VideoCapture(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        sucesso, frame = camera.read()

        if not sucesso:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = maos.process(rgb)

        gesto_atual = "nenhum"

        if resultado.multi_hand_landmarks:
            for mao in resultado.multi_hand_landmarks:
                mp_desenho.draw_landmarks(
                    frame,
                    mao,
                    mp_maos.HAND_CONNECTIONS
                )

                pontos = mao.landmark
                gesto_detectado = "nenhum"

                if eh_joinha(pontos):
                    gesto_detectado = "joinha"

                elif eh_hangloose(pontos):
                    gesto_detectado = "hangloose"

                elif eh_dedo_meio(pontos):
                    gesto_detectado = "dedo_meio"

                if gesto_detectado != "nenhum":
                    gesto_atual = gesto_detectado

                    if gesto_detectado != ultimo_gesto_salvo:
                        registrar(gesto_detectado)
                        salvar_captura(frame, gesto_detectado)

                        if gesto_detectado == "dedo_meio":
                            mostrar_imagem = True

                        ultimo_gesto_salvo = gesto_detectado

                    ultimo_gesto = gesto_detectado

                if gesto_detectado == "joinha":
                    cv2.putText(
                        frame,
                        "MAQUINA LIGADA",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3
                    )

                elif gesto_detectado == "hangloose":
                    cv2.putText(
                        frame,
                        "MAQUINA DESLIGADA",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3
                    )

                elif gesto_detectado == "dedo_meio":
                    cv2.putText(
                        frame,
                        "GESTO DETECTADO",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type:image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    camera.release()


@app.route("/video_feed")
def video_feed():
    return Response(
        gerar_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    global mostrar_imagem

    dados = {
        "gesto": gesto_atual,
        "maquina": "Ligada" if maquina_ligada else "Desligada",
        "ligada": maquina_ligada,
        "mostrar_imagem": mostrar_imagem,
        "historico": historico[-10:]
    }

    mostrar_imagem = False

    return jsonify(dados)


if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False
    )