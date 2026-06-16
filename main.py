import tensorflow as tf
import os

print("--- DIAGNÓSTICO DE HARDWARE TENSORFLOW ---")

# 1. Comprobar si TensorFlow tiene acceso a alguna GPU
dispositivos_gpu = tf.config.list_physical_devices('GPU')

if len(dispositivos_gpu) > 0:
    print(f"\n¡ÉXITO! Se detectaron {len(dispositivos_gpu)} GPU(s) disponibles.")
    for i, gpu in enumerate(dispositivos_gpu):
        print(f"  - GPU {i}: {gpu}")

    # Obtener el nombre comercial del dispositivo asignado por Keras
    nombre_gpu = tf.test.gpu_device_name()
    print(f"\nDispositivo activo para Keras: {nombre_gpu}")
else:
    print("\n⚠️ ADVERTENCIA: No se detectó ninguna GPU. TensorFlow está corriendo en CPU.")
    print("Si estás en Colab, ve a Entorno de ejecución -> Cambiar tipo de entorno -> T4 GPU.")

print("\n-----------------------------------------")

# 2. Comando de bajo nivel de NVIDIA (Solo funciona en Linux/Colab o entornos locales con drivers instalados)
# El comando de consola '!nvidia-smi' nos da el reporte de consumo de memoria y el modelo exacto en tiempo real.
try:
    print("Reporte oficial de la tarjeta de video (nvidia-smi):")
    print("====================================================")
    os.system('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv')
except:
    print("Comando nvidia-smi no disponible en este sistema operativo (común en Windows nativo sin WSL2 o macOS).")