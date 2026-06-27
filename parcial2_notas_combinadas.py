# -*- coding: utf-8 -*-
"""
=============================================================================
PARCIAL 2 – NOTAS DE CLASE COMBINADAS
Aprendizaje Automático II | Universidad Iberoamericana León
=============================================================================
CLASE 1: CNNs – Convolución, Transfer Learning y Grad-CAM
CLASE 2: RNNs – Series Temporales con LSTM, GRU y SimpleRNN
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS GENERALES
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Para la Clase 2 (RNNs con backend PyTorch en Keras 3)
os.environ["KERAS_BACKEND"] = "torch"
import torch


# =============================================================================
# CLASE 1: CNNs — CONVOLUCIÓN, TRANSFER LEARNING Y GRAD-CAM
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1.1  CONVOLUCIÓN 2D DESDE CERO
# ─────────────────────────────────────────────────────────────────────────────
def convolucion_2d(f, w):
    """Aplica convolución 2D sin padding, stride=1."""
    f_h, f_w = f.shape
    w_h, w_w = w.shape
    out = np.zeros((f_h - w_h + 1, f_w - w_w + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(f[i:i+w_h, j:j+w_w] * w)
    return np.clip(out, 0, 255)

# Kernels de ejemplo:
kernel_sharpen = np.array([[ 0,-1, 0],[-1, 5,-1],[ 0,-1, 0]])
kernel_blur    = np.ones((3,3)) / 9.0
kernel_sobel_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])


# ─────────────────────────────────────────────────────────────────────────────
# 1.2  API SECUENCIAL vs API FUNCIONAL
# ─────────────────────────────────────────────────────────────────────────────
# Sequential: pila lineal de capas, NO soporta skip connections (ej. ResNet).
# Funcional : grafo acíclico dirigido (DAG) → estándar de producción.

# Ejemplo API Funcional (patrón base para todo el parcial):
#   inputs  = keras.Input(shape=(H, W, C))
#   x       = layers.Conv2D(32, 3, activation="relu")(inputs)
#   outputs = layers.Dense(1, activation="sigmoid")(x)
#   model   = keras.Model(inputs=inputs, outputs=outputs)


# ─────────────────────────────────────────────────────────────────────────────
# 1.3  CALLBACKS PROFESIONALES
# ─────────────────────────────────────────────────────────────────────────────
def obtener_callbacks(nombre_modelo):
    """Devuelve EarlyStopping + ModelCheckpoint frescos por experimento."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"mejor_modelo_{nombre_modelo}.keras",
            save_best_only=True, monitor="val_loss"
        )
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1.4  DATASET: Cats vs Dogs (3 000 imágenes)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
IMG_SIZE   = (180, 180)

# Supone que el dataset ya está en PATH (descargado con kagglehub)
# train_dataset  = tf.keras.utils.image_dataset_from_directory(train_dir,  ...)
# val_dataset    = tf.keras.utils.image_dataset_from_directory(val_dir, ...)
# → Internamente: decodifica, redimensiona, lote y baraja on-the-fly.
# → Particionamos validation 50/50 para val_puro y test_dataset.


# ─────────────────────────────────────────────────────────────────────────────
# 1.5  CNN MANUAL BASE (sin regularización → overfitting esperado)
# ─────────────────────────────────────────────────────────────────────────────
def build_cnn_manual():
    inputs = keras.Input(shape=(180, 180, 3))
    x = layers.Rescaling(1./255)(inputs)           # Normalizar [0,255]→[0,1]
    x = layers.Conv2D(32,  3, activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,  3, activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="CNN_Manual")

# model_manual = build_cnn_manual()
# model_manual.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# history_manual = model_manual.fit(train_dataset, epochs=15,
#                                   validation_data=val_dataset,
#                                   callbacks=obtener_callbacks("manual"))


# ─────────────────────────────────────────────────────────────────────────────
# 1.6  CNN REGULARIZADA: Data Augmentation + Dropout
# ─────────────────────────────────────────────────────────────────────────────
# Data Augmentation: genera variedad sintética → rompe memorización por píxel.
# Solo se activa durante model.fit(); se apaga en evaluate/predict.
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="DataAugmentation")

def build_cnn_regularizada():
    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)                  # Augmentation primero
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(32,  3, activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,  3, activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x); x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)                    # Apaga 50% nodos → anti-overfitting
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="CNN_Regularizada")


# ─────────────────────────────────────────────────────────────────────────────
# 1.7  TRANSFER LEARNING (Feature Extraction) — Fase 1
# ─────────────────────────────────────────────────────────────────────────────
# Concepto: reutilizar backbone preentrenado en ImageNet (millones de imágenes).
# Fase 1: base_model.trainable = False → solo entrena la cabeza nueva.
# GlobalAveragePooling2D en lugar de Flatten → evita explosión de parámetros.

def build_transfer_model(base_arch="ResNet50V2"):
    base_model = keras.applications.ResNet50V2(
        weights="imagenet", include_top=False, input_shape=(180, 180, 3)
    )
    base_model.trainable = False                   # Congelar pesos del backbone
    preprocess = keras.applications.resnet_v2.preprocess_input

    inputs = keras.Input(shape=(180, 180, 3))
    x = data_augmentation(inputs)
    x = preprocess(x)
    x = base_model(x, training=False)             # training=False → BatchNorm en inferencia
    x = layers.GlobalAveragePooling2D()(x)        # Compacta volumen 3D → vector
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=f"TransferLearning_{base_arch}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.8  INFERENCIA: regla del expand_dims
# ─────────────────────────────────────────────────────────────────────────────
# model.predict() exige lote → pasar (1, H, W, 3) aunque sea 1 imagen:
#   img_batch = tf.expand_dims(img_individual, axis=0)
#   prob = model.predict(img_batch, verbose=0)[0][0]
#   clase = "Dog" if prob > 0.5 else "Cat"


# ─────────────────────────────────────────────────────────────────────────────
# 1.9  XAI: GRAD-CAM (Gradient-weighted Class Activation Mapping)
# ─────────────────────────────────────────────────────────────────────────────
# Idea: ¿qué zona de la imagen fue clave para la predicción?
# 1. Tomar la última capa Conv (todavía tiene info espacial).
# 2. Gradiente del score respecto a sus activaciones → importancia por canal.
# 3. Suma ponderada + ReLU → heatmap → superponer a imagen original.

def visualizar_gradcam(modelo, nombre_capa, img_array):
    """
    Genera y muestra el mapa Grad-CAM para la capa indicada.
    img_array: np.array de forma (1, H, W, 3).
    """
    modelo.eval()
    modelo.zero_grad()

    # Sub-modelo que expone la capa convolucional y la salida final
    grad_model = keras.Model(
        inputs=modelo.inputs,
        outputs=[modelo.get_layer(nombre_capa).output, modelo.output]
    )

    # Forward pass con rastreo de gradientes (PyTorch backend)
    input_tensor = torch.tensor(img_array, dtype=torch.float32)
    input_tensor.requires_grad = True
    conv_out, pred = grad_model(input_tensor, training=False)
    conv_out.retain_grad()
    pred[:, 0].backward()

    # Heatmap = suma ponderada de activaciones por gradiente promediado
    weights  = torch.mean(conv_out.grad, dim=(0, 1, 2))
    heatmap  = torch.relu(torch.sum(weights * conv_out, dim=-1))
    heatmap  = heatmap / (torch.max(heatmap) + 1e-8)
    heatmap_np = heatmap.squeeze().detach().cpu().numpy()

    # Superponer heatmap coloreado sobre imagen original
    h, w = img_array.shape[1], img_array.shape[2]
    heatmap_r = cv2.resize(heatmap_np, (w, h))
    heatmap_c = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * heatmap_r), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )
    img_orig   = img_array[0].astype("uint8")
    superpos   = cv2.addWeighted(img_orig, 0.6, heatmap_c, 0.4, 0)

    score    = pred[:, 0].item()
    etiqueta = "Perro" if score > 0.5 else "Gato"
    confianza = score if score > 0.5 else (1 - score)

    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1); plt.imshow(img_orig);   plt.title("Original"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(superpos)
    plt.title(f"Grad-CAM → {etiqueta} ({confianza*100:.1f}%)")
    plt.axis("off"); plt.tight_layout(); plt.show()


# =============================================================================
# CLASE 2: RNNs — SERIES TEMPORALES (DATASET JENA CLIMATE)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  CARGA Y EXPLORACIÓN DEL DATASET (Jena Climate 2009-2016)
# ─────────────────────────────────────────────────────────────────────────────
# import pandas as pd, requests, zipfile, io
# df = pd.read_csv("jena_climate_2009_2016.csv")
# Registros cada 10 min → ~420 K filas, 14 variables meteorológicas.


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  PREPARACIÓN: DIVISIÓN CRONOLÓGICA Y NORMALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
# ¡NO hacer shuffle global! Destruiría la causalidad temporal.
# División: Train 50% | Val 25% | Test 25% (en orden de tiempo).
# Normalización: media y std calculados SOLO con train → aplicados a los 3 sets.
# Esto evita Data Leakage.

# target    = df["T (degC)"].values          # Variable a predecir
# raw_data  = df.drop("Date Time", axis=1).values
# mean, std = raw_data[:n_train].mean(axis=0), raw_data[:n_train].std(axis=0)
# raw_data  = (raw_data - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  GENERADOR DE VENTANAS DESLIZANTES
# ─────────────────────────────────────────────────────────────────────────────
# keras.utils.timeseries_dataset_from_array:
#   sampling_rate   = 6   → 1 muestra/hora (original: cada 10 min)
#   sequence_length = 120 → ventana de 5 días hacia el pasado
#   delay = 6 * (120 + 24 - 1) → predecir 24 h en el futuro
# La función crea un tf.data.Dataset eficiente (no carga todo en RAM).


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  BASELINE DE SENTIDO COMÚN
# ─────────────────────────────────────────────────────────────────────────────
# Predicción naïve: "la temperatura de mañana = la de ahora"
# Sirve como cota inferior que cualquier modelo debe superar.


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  MLP y CNN 1D (modelos estáticos de referencia)
# ─────────────────────────────────────────────────────────────────────────────
def build_mlp_temporal():
    return models.Sequential([
        layers.Flatten(input_shape=(120, 14)),    # Aplana toda la ventana
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ], name="MLP_Temporal")

def build_cnn1d():
    return models.Sequential([
        layers.Conv1D(8, 24, activation="relu", input_shape=(120, 14)),
        layers.MaxPooling1D(2),
        layers.Conv1D(8, 12, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(1)
    ], name="CNN_1D")

# Ambos se compilan con: optimizer="rmsprop", loss="mse", metrics=["mae"]
# MAE real (°C) = mae_normalizado * std_target


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  SIMPLERNN
# ─────────────────────────────────────────────────────────────────────────────
# Ecuación: h_t = tanh(W·x_t + U·h_{t-1} + b)
# Problema: vanishing gradient → olvida el pasado lejano (largo plazo).

def build_simple_rnn():
    return models.Sequential([
        layers.SimpleRNN(16, input_shape=(120, 14)),
        layers.Dense(1)
    ], name="SimpleRNN")


# ─────────────────────────────────────────────────────────────────────────────
# 2.7  LSTM — Long Short-Term Memory
# ─────────────────────────────────────────────────────────────────────────────
# Solución al vanishing gradient mediante Estado de Celda (C_t) aditivo.
# 3 compuertas:
#   Olvido  (f_t): descarta info pasada del estado de celda.
#   Entrada (i_t): decide qué nueva info almacenar.
#   Salida  (o_t): filtra qué parte del estado de celda se convierte en h_t.
# C_t = f_t ⊙ C_{t-1}  +  i_t ⊙ tanh(W_c·x_t + U_c·h_{t-1} + b_c)

def build_lstm():
    return models.Sequential([
        layers.LSTM(32, input_shape=(120, 14)),
        layers.Dense(1)
    ], name="LSTM")

def build_lstm_dropout():
    """LSTM con Recurrent Dropout (Gal & Ghahramani, 2016).
    Aplica la MISMA máscara de dropout en todos los pasos temporales
    para no romper la memoria de largo plazo."""
    return models.Sequential([
        layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2,
                    input_shape=(120, 14)),
        layers.Dense(1)
    ], name="LSTM_Dropout")


# ─────────────────────────────────────────────────────────────────────────────
# 2.8  GRU APILADO (Stacked GRU)
# ─────────────────────────────────────────────────────────────────────────────
# GRU = simplificación de LSTM: sin estado de celda separado, 2 compuertas
# (Reset r_t y Update z_t) → ~33% menos parámetros que LSTM.
#
# Para apilar capas recurrentes, la capa intermedia necesita return_sequences=True
# para devolver tensor 3D (Batch, Timesteps, Units) a la siguiente capa.

def build_stacked_gru():
    return models.Sequential([
        layers.GRU(32, recurrent_dropout=0.5, return_sequences=True,
                   input_shape=(120, 14), name="GRU_Intermedia"),
        layers.GRU(16, recurrent_dropout=0.5, return_sequences=False,
                   name="GRU_Final"),
        layers.Dense(1, name="Prediccion")
    ], name="StackedGRU")


# ─────────────────────────────────────────────────────────────────────────────
# 2.9  UTILIDADES: ENTRENAMIENTO Y EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def compilar_y_entrenar(model, train_ds, val_ds, epochs=10, nombre="modelo"):
    """Compila con rmsprop+mse y entrena guardando el mejor checkpoint."""
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    cb = [keras.callbacks.ModelCheckpoint(f"jena_{nombre}.keras",
                                          save_best_only=True)]
    history = model.fit(train_ds, epochs=epochs,
                        validation_data=val_ds, callbacks=cb, verbose=1)
    return history

def mae_celsius(model, test_ds, std_target):
    """Devuelve el MAE desnormalizado en °C."""
    mae_norm = model.evaluate(test_ds, verbose=0)[1]
    return mae_norm * std_target

def plot_mae(history, titulo="Modelo"):
    """Grafica Training vs Validation MAE."""
    mae     = history.history["mae"]
    val_mae = history.history["val_mae"]
    plt.figure()
    plt.plot(range(1, len(mae)+1), mae,     "r--", label="Train MAE")
    plt.plot(range(1, len(mae)+1), val_mae, "b",   label="Val MAE")
    plt.title(f"MAE — {titulo}")
    plt.legend(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 2.10  TABLA COMPARATIVA DE RESULTADOS (esquema de uso)
# ─────────────────────────────────────────────────────────────────────────────
# Al finalizar el entrenamiento de todos los modelos:
#
# print("Modelo           | MAE (°C)")
# print("-----------------+---------")
# print(f"Sentido Común    | {mae_sc:.2f}")
# print(f"MLP              | {mae_mlp:.2f}")
# print(f"CNN 1D           | {mae_cnn:.2f}")
# print(f"SimpleRNN        | {mae_rnn:.2f}")
# print(f"LSTM             | {mae_lstm:.2f}")
# print(f"LSTM + Dropout   | {mae_lstm_drop:.2f}")
# print(f"Stacked GRU      | {mae_gru:.2f}")


# =============================================================================
# RESUMEN CONCEPTUAL RÁPIDO
# =============================================================================
"""
CLASE 1 — CNNs:
  • Convolución 2D: desliza un kernel sobre la imagen, extrae rasgos locales.
  • API Funcional > Sequential para modelos con skip connections (ResNet).
  • CNN Manual → overfitting → solucionamos con Data Augmentation + Dropout.
  • Transfer Learning: backbone congelado (ImageNet) + nueva cabeza sigmoid.
  • GlobalAveragePooling2D: compacta sin explotar parámetros (mejor que Flatten).
  • Grad-CAM: heatmap que muestra QUÉ zonas activan la predicción.

CLASE 2 — RNNs:
  • División temporal 50/25/25 y normalización solo con train.
  • timeseries_dataset_from_array: ventanas deslizantes eficientes.
  • SimpleRNN → vanishing gradient → olvida dependencias largas.
  • LSTM: estado de celda (C_t) aditivo → preserva info a largo plazo.
  • Recurrent Dropout: misma máscara en cada paso temporal → regularización
    correcta en RNNs (no el Dropout estándar).
  • GRU: versión eficiente de LSTM con 2 compuertas, menos parámetros.
  • Stacked RNN: return_sequences=True en capas intermedias.
"""
