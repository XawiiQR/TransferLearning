
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
