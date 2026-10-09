"""
Visualizador interactivo de una CNN sobre MNIST.

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import traceback
from typing import Optional

# Reduce ruido de TensorFlow antes de importarlo.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as exc:  # pragma: no cover - entorno sin TF
    st.error(
        "No se pudo importar TensorFlow/Keras. Instala las dependencias "
        "con `pip install -r requirements.txt`."
    )
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------------------------
# Constantes de arquitectura y entrenamiento
# ---------------------------------------------------------------------------
IMG_SIZE = 28
N_CLASSES = 10
CONV1_FILTERS = 32
CONV2_FILTERS = 64
DENSE_UNITS = 128
KERNEL_SIZE = 3
POOL_SIZE = 2
TRAIN_SAMPLES = 12_000
EPOCHS = 2
BATCH_SIZE = 256
WEIGHTS_PATH = "cnn_mnist_weights.keras"
RANDOM_SEED = 42

LAYER_CONV1 = "conv1"
LAYER_POOL1 = "pool1"
LAYER_CONV2 = "conv2"
LAYER_POOL2 = "pool2"
LAYER_FLAT = "flatten"
LAYER_DENSE = "dense_hidden"
LAYER_OUTPUT = "output"


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Fija semillas para que el entrenamiento de demostración sea reproducible."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_cnn() -> keras.Model:
    """Construye la CNN pedida: Conv-Pool-Conv-Pool-Flatten-MLP-Softmax."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="entrada")

    x = layers.Conv2D(
        CONV1_FILTERS,
        KERNEL_SIZE,
        activation="relu",
        padding="valid",
        name=LAYER_CONV1,
    )(inputs)
    x = layers.MaxPooling2D(POOL_SIZE, name=LAYER_POOL1)(x)

    x = layers.Conv2D(
        CONV2_FILTERS,
        KERNEL_SIZE,
        activation="relu",
        padding="valid",
        name=LAYER_CONV2,
    )(x)
    x = layers.MaxPooling2D(POOL_SIZE, name=LAYER_POOL2)(x)

    x = layers.Flatten(name=LAYER_FLAT)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name=LAYER_DENSE)(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax", name=LAYER_OUTPUT)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_mnist_demo")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _prepare_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Descarga MNIST y lo normaliza a [0, 1] con canal de profundidad 1."""
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as exc:
        raise RuntimeError(
            "No se pudo descargar o cargar el dataset MNIST. "
            "Comprueba la conexión a internet e inténtalo de nuevo."
        ) from exc

    if x_train.size == 0 or x_test.size == 0:
        raise RuntimeError("MNIST se cargó vacío. Vuelve a ejecutar la aplicación.")

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test


@st.cache_resource(show_spinner="Cargando MNIST y preparando la CNN (solo la primera vez)...")
def load_model_and_data() -> tuple[keras.Model, np.ndarray, np.ndarray, dict]:
    """Carga pesos si existen; si no, entrena un subconjunto de MNIST (1-2 épocas)."""
    set_seeds()
    x_train, y_train, x_test, y_test = _prepare_mnist()

    n_train = min(TRAIN_SAMPLES, len(x_train))
    x_small = x_train[:n_train]
    y_small = y_train[:n_train]

    model = build_cnn()
    info: dict = {
        "trained_now": False,
        "train_samples": n_train,
        "epochs": EPOCHS,
        "train_acc": None,
        "val_acc": None,
    }

    if os.path.exists(WEIGHTS_PATH):
        try:
            model.load_weights(WEIGHTS_PATH)
            info["source"] = "pesos locales"
            return model, x_test, y_test, info
        except Exception:
            # Si los pesos no coinciden con la arquitectura, se reentrena.
            model = build_cnn()

    history = model.fit(
        x_small,
        y_small,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test[:2000], y_test[:2000]),
        verbose=0,
    )
    try:
        model.save_weights(WEIGHTS_PATH)
    except Exception:
        pass

    info["trained_now"] = True
    info["source"] = "entrenamiento rápido"
    info["train_acc"] = float(history.history["accuracy"][-1])
    info["val_acc"] = float(history.history["val_accuracy"][-1])
    return model, x_test, y_test, info


def build_activation_model(model: keras.Model) -> keras.Model:
    """Modelo auxiliar que expone las activaciones de cada capa intermedia."""
    layer_names = [
        LAYER_CONV1,
        LAYER_POOL1,
        LAYER_CONV2,
        LAYER_POOL2,
        LAYER_FLAT,
        LAYER_DENSE,
        LAYER_OUTPUT,
    ]
    outputs = [model.get_layer(name).output for name in layer_names]
    return keras.Model(inputs=model.inputs, outputs=outputs, name="cnn_activations")


def preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """Convierte una imagen subida a un tensor MNIST: 28x28x1, fondo oscuro, [0, 1]."""
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = ImageOps.exif_transpose(image)

    # Recorte centrado a cuadrado para no distorsionar el dígito.
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    array = np.asarray(image).astype("float32") / 255.0
    # MNIST es dígito claro sobre fondo oscuro. Si el fondo es claro, invertimos.
    if array.mean() > 0.5:
        array = 1.0 - array

    # Un umbral suave ayuda a imágenes con fondo grisáceo.
    array = np.clip((array - 0.15) / 0.85, 0.0, 1.0)
    return array.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def get_activations(activation_model: keras.Model, image: np.ndarray) -> dict[str, np.ndarray]:
    """Pasa una imagen por la CNN y devuelve las activaciones de cada capa."""
    outputs = activation_model.predict(image, verbose=0)
    names = [
        LAYER_CONV1,
        LAYER_POOL1,
        LAYER_CONV2,
        LAYER_POOL2,
        LAYER_FLAT,
        LAYER_DENSE,
        LAYER_OUTPUT,
    ]
    return {name: out[0] for name, out in zip(names, outputs)}


def _style_axes(ax: plt.Axes, title: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, pad=2)


def plot_input_image(image: np.ndarray, true_label: Optional[int]) -> plt.Figure:
    """Muestra la imagen 28x28 ampliada."""
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    title = "28×28 × 1 canal"
    if true_label is not None:
        title = f"Etiqueta real: {true_label}  ·  {title}"
    ax.set_title(title, fontsize=11)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_feature_maps(
    maps: np.ndarray,
    title: str,
    max_maps: Optional[int] = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """Dibuja una cuadrícula de mapas de características (H, W, C)."""
    height, width, n_channels = maps.shape
    n_show = n_channels if max_maps is None else min(max_maps, n_channels)
    cols = 8
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.25))
    axes = np.atleast_1d(axes).ravel()

    for i in range(rows * cols):
        ax = axes[i]
        if i < n_show:
            ax.imshow(maps[:, :, i], cmap=cmap, interpolation="nearest")
            _style_axes(ax, f"f{i}")
        else:
            ax.axis("off")

    fig.suptitle(f"{title}  ·  {height}×{width}×{n_channels}", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


def plot_flatten_vector(vector: np.ndarray) -> plt.Figure:
    """Representa el tensor aplanado como una tira de calor 1D y un recorte numérico."""
    vec = vector.reshape(-1)
    n = vec.size

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 3.4),
        gridspec_kw={"height_ratios": [1.1, 2.2]},
    )

    axes[0].imshow(vec.reshape(1, -1), aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_yticks([])
    axes[0].set_xlabel(f"Índice del vector ({n} valores)")
    axes[0].set_title("Flatten: tensor 5×5×64 → vector 1D de 1.600 activaciones")

    preview = min(128, n)
    axes[1].plot(np.arange(preview), vec[:preview], color="#2E86AB", linewidth=1.0)
    axes[1].fill_between(np.arange(preview), vec[:preview], alpha=0.25, color="#2E86AB")
    axes[1].set_xlim(0, preview - 1)
    axes[1].set_xlabel(f"Primeros {preview} valores del vector")
    axes[1].set_ylabel("Activación")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_dense_and_prediction(
    dense_acts: np.ndarray,
    probs: np.ndarray,
    true_label: Optional[int],
) -> tuple[plt.Figure, plt.Figure, int, float]:
    """Visualiza la capa densa oculta y las probabilidades softmax."""
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    fig_dense, ax_d = plt.subplots(figsize=(10, 2.2))
    ax_d.imshow(dense_acts.reshape(1, -1), aspect="auto", cmap="magma", interpolation="nearest")
    ax_d.set_yticks([])
    ax_d.set_xlabel("Neuronas de la capa densa oculta (128, ReLU)")
    ax_d.set_title("MLP — activaciones de la capa oculta")
    fig_dense.tight_layout()

    colors = ["#2ecc71" if i == pred else "#95a5a6" for i in range(N_CLASSES)]
    fig_pred, ax_p = plt.subplots(figsize=(8, 4.2))
    bars = ax_p.bar(range(N_CLASSES), probs, color=colors, edgecolor="black", linewidth=0.4)
    ax_p.set_xticks(range(N_CLASSES))
    ax_p.set_xlabel("Dígito")
    ax_p.set_ylabel("Probabilidad (softmax)")
    ax_p.set_ylim(0, 1.08)
    ax_p.set_title("Predicción final de la CNN")
    for bar, p in zip(bars, probs):
        ax_p.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{p:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if true_label is not None:
        ax_p.axvline(true_label, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.8)
        ax_p.legend(
            [bars[pred], plt.Line2D([0], [0], color="#e74c3c", linestyle="--")],
            ["Predicción", "Etiqueta real"],
            loc="upper right",
        )
    ax_p.grid(True, axis="y", alpha=0.3)
    fig_pred.tight_layout()
    return fig_dense, fig_pred, pred, confidence


def render_sidebar() -> tuple[str, Optional[int], Optional[bytes]]:
    """Controles de la barra lateral: origen de la imagen e índice MNIST."""
    st.sidebar.title("CNN paso a paso")
    st.sidebar.markdown(
        "Explora cómo una **red neuronal convolucional** transforma un dígito "
        "manuscrito (MNIST) hasta la predicción final."
    )
    st.sidebar.markdown(
        f"""
**Arquitectura**
- Entrada: `{IMG_SIZE}×{IMG_SIZE}×1`
- Conv2D `{CONV1_FILTERS}` filtros 3×3 + ReLU → 26×26
- MaxPool 2×2 → 13×13
- Conv2D `{CONV2_FILTERS}` filtros 3×3 + ReLU → 11×11
- MaxPool 2×2 → 5×5
- Flatten → 1600
- Dense `{DENSE_UNITS}` + ReLU
- Dense 10 + softmax
        """
    )

    source = st.sidebar.radio(
        "Origen de la imagen",
        ("Muestra de MNIST", "Subir imagen"),
        index=0,
    )

    index: Optional[int] = None
    uploaded_bytes: Optional[bytes] = None

    if source == "Muestra de MNIST":
        if st.sidebar.button("Imagen aleatoria", use_container_width=True):
            st.session_state["mnist_index"] = int(np.random.randint(0, 10_000))
        default_idx = int(st.session_state.get("mnist_index", 0))
        index = st.sidebar.slider("Índice del conjunto de prueba", 0, 9999, default_idx)
        st.session_state["mnist_index"] = index
    else:
        uploaded = st.sidebar.file_uploader(
            "Imagen manuscrita (PNG/JPG). Se redimensiona a 28×28.",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()

    st.sidebar.caption(
        "Las imágenes subidas se convierten a escala de grises. "
        "Si el dígito es oscuro sobre fondo claro, se invierte para imitar MNIST."
    )
    return source, index, uploaded_bytes


def main() -> None:
    st.set_page_config(
        page_title="CNN MNIST — Visualización capa a capa",
        page_icon="🧠",
        layout="wide",
    )
    st.title("Cómo ve una CNN un dígito de MNIST")
    st.markdown(
        "Selecciona o sube un dígito y recorre **convolución, pooling, aplanado y MLP**. "
        "Cada mapa de características es la activación real de un filtro de la red."
    )

    source, index, uploaded_bytes = render_sidebar()

    try:
        model, x_test, y_test, info = load_model_and_data()
    except Exception as exc:
        st.error("No se pudo cargar o entrenar el modelo.")
        st.code("".join(traceback.format_exception_only(type(exc), exc)))
        st.stop()

    with st.expander("Estado del modelo", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Origen", info.get("source", "—"))
        cols[1].metric("Muestras de entrenamiento", f"{info.get('train_samples', 0):,}")
        cols[2].metric("Épocas", info.get("epochs", "—"))
        val_acc = info.get("val_acc")
        cols[3].metric(
            "Accuracy validación",
            f"{val_acc:.1%}" if val_acc is not None else "pesos cargados",
        )
        st.caption(
            "El entrenamiento usa un subconjunto de MNIST (2 épocas) para que la demo "
            "arranque en segundos. Los pesos se guardan en `cnn_mnist_weights.keras`."
        )

    true_label: Optional[int] = None
    image: Optional[np.ndarray] = None

    try:
        if source == "Muestra de MNIST":
            idx = 0 if index is None else int(index)
            idx = max(0, min(idx, len(x_test) - 1))
            image = x_test[idx : idx + 1]
            true_label = int(y_test[idx])
        else:
            if uploaded_bytes is None:
                st.info("Sube una imagen en la barra lateral para comenzar.")
                st.stop()
            image = preprocess_uploaded_image(uploaded_bytes)
    except Exception as exc:
        st.error("No se pudo preparar la imagen de entrada.")
        st.exception(exc)
        st.stop()

    activation_model = build_activation_model(model)
    try:
        acts = get_activations(activation_model, image)
    except Exception as exc:
        st.error("Falló la extracción de activaciones intermedias.")
        st.exception(exc)
        st.stop()

    st.header("Imagen de entrada")
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.pyplot(plot_input_image(image, true_label), use_container_width=False)
    with col_info:
        st.markdown(
            """
La red recibe un tensor **28×28×1** con valores en `[0, 1]`.  
Cada píxel es una intensidad: en MNIST el trazo es claro y el fondo es negro.

A partir de aquí, los **filtros 3×3** buscan bordes, curvas y texturas locales.  
El **pooling** reduce el tamaño espacial y conserva las activaciones más fuertes.
            """
        )

    st.divider()
    st.header("Paso 1 — Primera convolución (3×3, 32 filtros, ReLU)")
    st.markdown(
        "El filtro 3×3 recorre la imagen y produce **32 mapas de 26×26** "
        "(28 − 3 + 1). Cada mapa destaca un patrón distinto: trazos verticales, "
        "curvas o esquinas. El color indica la intensidad de la activación ReLU."
    )
    st.pyplot(plot_feature_maps(acts[LAYER_CONV1], "Conv1 — mapas 26×26"), use_container_width=True)

    st.divider()
    st.header("Paso 2 — Primer Max Pooling (2×2)")
    st.markdown(
        "En cada ventana 2×2 se conserva **el valor máximo**. El mapa pasa de "
        "**26×26 a 13×13**: se reduce el detalle fino, pero se mantiene la señal "
        "más fuerte de cada zona (invarianza local a pequeñas traslaciones)."
    )
    st.pyplot(plot_feature_maps(acts[LAYER_POOL1], "Pool1 — mapas 13×13"), use_container_width=True)

    st.divider()
    st.header("Paso 3 — Segunda convolución (3×3, 64 filtros, ReLU)")
    st.markdown(
        "Sobre los 32 canales anteriores se aplican **64 filtros 3×3**. El resultado "
        "son mapas **11×11** más abstractos: combinan bordes en formas propias del dígito. "
        "Suelen ser difíciles de interpretar a simple vista."
    )
    st.pyplot(plot_feature_maps(acts[LAYER_CONV2], "Conv2 — mapas 11×11"), use_container_width=True)

    st.divider()
    st.header("Paso 4 — Segundo Max Pooling (2×2)")
    st.markdown(
        "Nueva reducción espacial: **11×11 → 5×5** (el resto de la división se descarta "
        "con `valid`). Queda un tensor compacto **5×5×64** listo para el clasificador."
    )
    st.pyplot(plot_feature_maps(acts[LAYER_POOL2], "Pool2 — mapas 5×5"), use_container_width=True)

    st.divider()
    st.header("Paso 5 — Aplanado (Flatten)")
    st.markdown(
        "El tensor 5×5×64 se convierte en un **vector de 1.600 números**. "
        "Esa tira es lo que alimenta al MLP: ya no hay estructura espacial explícita."
    )
    st.pyplot(plot_flatten_vector(acts[LAYER_FLAT]), use_container_width=True)

    st.divider()
    st.header("Paso 6 — MLP y predicción final (softmax)")
    st.markdown(
        "Una capa densa de **128 neuronas ReLU** mezcla las 1.600 activaciones. "
        "La capa de salida con **10 neuronas softmax** asigna una probabilidad a cada dígito 0–9."
    )
    fig_dense, fig_pred, pred, confidence = plot_dense_and_prediction(
        acts[LAYER_DENSE], acts[LAYER_OUTPUT], true_label
    )
    st.pyplot(fig_dense, use_container_width=True)
    st.pyplot(fig_pred, use_container_width=True)

    if true_label is not None:
        if pred == true_label:
            st.success(
                f"Predicción **{pred}** con confianza **{confidence:.1%}**. "
                f"Coincide con la etiqueta real ({true_label})."
            )
        else:
            st.error(
                f"Predicción **{pred}** con confianza **{confidence:.1%}**. "
                f"La etiqueta real es **{true_label}**."
            )
    else:
        st.info(f"Predicción **{pred}** con confianza **{confidence:.1%}** (imagen subida, sin etiqueta).")

    plt.close("all")


if __name__ == "__main__":
    main()
