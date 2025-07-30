
#  TransferLearning - Segmentación de Prendas con Transfer Learning usando Mask R-CNN y DeepFashion2

---

## ❓ Planteamiento del Problema

En el campo de la visión por computadora, una de las tareas más relevantes para la industria de la moda y el comercio electrónico es la **segmentación precisa de prendas de vestir**. Sin embargo, los modelos tradicionales de detección de objetos suelen limitarse a la predicción de cajas delimitadoras (*bounding boxes*), sin capturar el contorno real de la prenda ni su forma pixelada.

Esto representa una gran limitación en aplicaciones donde se requiere:

- **Separar con precisión distintas partes del cuerpo humano según la prenda**
- **Clasificar prendas con mayor semántica que simplemente "ropa"**
- **Extraer máscaras que permitan aplicar filtros, reconocimiento o comparación visual**

Por ello, surge la necesidad de construir un sistema de segmentación por instancia que no solo detecte prendas, sino que identifique y **separe de manera precisa la parte superior (*upper*) y la parte inferior (*lower*)** del atuendo de una persona, a nivel de píxel.

---

## 🎯 Objetivos del Proyecto

### Objetivo General

> Desarrollar un sistema de segmentación semántica de prendas de vestir utilizando *Transfer Learning* con Mask R-CNN, que permita detectar y diferenciar la **parte superior** y la **parte inferior** de la ropa en imágenes del dataset DeepFashion2.

---

### Objetivos Específicos

1. **Aplicar técnicas de Transfer Learning** sobre un modelo Mask R-CNN preentrenado para adaptarlo a un nuevo conjunto de clases: `upper` y `lower`.

2. **Preparar y reestructurar el dataset DeepFashion2** para su uso en PyTorch, creando una división clara en `train`, `val` y `test`, y simplificando las clases originales en dos macrocategorías semánticas.

3. **Construir una clase `Dataset` personalizada** que permita leer las anotaciones en formato JSON, extraer máscaras por píxel a partir de polígonos y generar etiquetas compatibles con Mask R-CNN.

4. **Entrenar y validar el modelo modificado**, utilizando métricas de pérdida y segmentación, comparando el desempeño con diferentes tamaños de datos y épocas de entrenamiento.

5. **Visualizar cualitativamente las predicciones del modelo**, mostrando las máscaras segmentadas y las etiquetas proyectadas sobre imágenes reales.


## 🧪 Preprocesamiento y Preparación del Dataset DeepFashion2

El dataset DeepFashion2, si bien es uno de los más completos para tareas de moda, no viene listo para usarse directamente con modelos como Mask R-CNN. A continuación, se detallan todas las transformaciones realizadas para dejarlo en un formato óptimo para el entrenamiento.

---

### 📁 1. Reorganización de Carpeta y Estructura de Datos

El dataset original fue reorganizado en una estructura tipo COCO con divisiones explícitas:

```
deepfashion2_coco/
├── train/
│   ├── images/
│   └── annos/
├── val/
│   ├── images/
│   └── annos/
└── test/
    ├── images/
    └── annos/
```

> 📌 **Código clave**: Se utilizó `shutil.copy()` para mover imágenes y anotaciones con nombres coincidentes (`image.jpg` y `image.json`).

---

### 🔍 2. Filtrado de Muestras Válidas

- Se conservaron solo imágenes que **tenían su anotación JSON correspondiente**.
- Se aplicó un **límite de imágenes** (`max_images`) para permitir pruebas rápidas.
- Las imágenes fueron barajadas aleatoriamente con `random.shuffle()` para garantizar aleatoriedad en la división.

```python
image_files = list(images_path.glob("*.jpg"))
random.seed(42)
random.shuffle(image_files)
```

---

### 🧾 3. Reducción de Categorías a Upper y Lower

Se creó un `category_map` que agrupa las 13 categorías originales en solo 2 clases semánticas:

```python
category_map = {
    "short sleeve top": "upper",
    "long sleeve top": "upper",
    "short sleeve outwear": "upper",
    "long sleeve outwear": "upper",
    "vest": "upper",
    "sling": "upper",
    "dress": "upper",
    "shorts": "lower",
    "trousers": "lower",
    "skirt": "lower"
}
```

- Las clases fueron reetiquetadas internamente como: `1 = upper`, `2 = lower`

---

### 🧠 4. Creación de Máscaras Binarias

Cada polígono de segmentación del `.json` fue convertido a una máscara por píxel usando OpenCV:

```python
mask = np.zeros((height, width), dtype=np.uint8)
polygon = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
cv2.fillPoly(mask, [polygon], color=1)
```

Esto permitió que el modelo aprendiera **qué píxeles pertenecen a la prenda** y cuáles no.

---

### 🧰 5. Dataset Personalizado para PyTorch

Se implementó la clase `DeepFashionDataset`, compatible con `torch.utils.data.Dataset`. Esta clase:

- Carga imágenes con `PIL.Image`
- Lee los `.json` y convierte cajas, etiquetas y máscaras a tensores
- Aplica transformaciones (`ToTensor`) para normalizar las entradas

```python
image, target = dataset[idx]
target = {
    "boxes": Tensor[n, 4],
    "labels": Tensor[n],
    "masks": Tensor[n, H, W],
    "image_id": Tensor[1]
}
```

> 📌 Esta clase fue usada tanto para entrenamiento como para validación, permitiendo flexibilidad total.

---

### ⚠️ 6. Control de Errores y Datos Vacíos

- Se ignoraron objetos sin `bounding_box` o sin `segmentation` válida.
- Se devolvieron tensores vacíos si la imagen no contenía ninguna instancia válida.
- Esto evitó errores en tiempo de entrenamiento con el modelo Mask R-CNN.

---

### 👁️ 7. Visualización de Muestras Anotadas

Se implementó una función `show_random_samples()` para inspección visual del dataset:

- Dibuja las `bounding boxes` en verde
- Superpone las máscaras en rojo con transparencia
- Muestra la etiqueta (`upper` o `lower`) sobre cada objeto detectado

> 📸 **Recomendación**: Inserta capturas de esta función en acción. Puedes usar una diapositiva como:  
> “Imagen segmentada desde el dataset” del PDF de presentación.

---

### ✅ Estado Final del Dataset

- Dataset dividido, filtrado y validado
- Máscaras por píxel listas para segmentación
- Etiquetas reducidas a `upper` y `lower`
- Dataset 100% compatible con `Mask R-CNN`


