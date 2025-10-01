// Perceptron para Clasificación de Riesgo Académico
// Ejecutar en Node.js: node perceptron_riesgo.js

// ==============================
// 1) Funciones de activación
// ==============================
function sigmoid(x) {
    // proteger frente a overflow
    if (x < -500) return 0;
    if (x > 500) return 1;
    return 1 / (1 + Math.exp(-x));
  }
  
  function activation(name, x) {
    switch (name) {
      case "linear": return x;
      case "step": return x >= 0 ? 1 : 0;
      case "sigmoid": return sigmoid(x);
      case "relu": return Math.max(0, x);
      case "tanh": return Math.tanh(x);
      case "softmax":
        // En un perceptrón de salida única, softmax se reduce a una probabilidad.
        // Aquí lo adaptamos devolviendo la sigmoide (equivalente para binario).
        return sigmoid(x);
      default: return x;
    }
  }
  
  // ==============================
  // 2) Clase Perceptrón (desde cero)
  // ==============================
  class Perceptron {
    constructor(inputSize, activationName = "sigmoid", lr = 0.1) {
      this.w = Array.from({length: inputSize}, () => Math.random() * 2 - 1); // pesos aleatorios
      this.b = Math.random() * 2 - 1; // bias (sesgo)
      this.lr = lr;
      this.activationName = activationName;
    }
  
    net(x) {
      return this.w.reduce((s, wi, i) => s + wi * x[i], this.b);
    }
  
    predict(x) {
      const z = this.net(x);
      return activation(this.activationName, z);
    }
  
    train(X, y, epochs = 20) {
      for (let e = 0; e < epochs; e++) {
        for (let i = 0; i < X.length; i++) {
          const xi = X[i];
          const yi = y[i];
          const yPred = this.predict(xi);
          const error = yi - yPred;
          // regla de aprendizaje del perceptrón (versión supervisada simple)
          for (let j = 0; j < this.w.length; j++) {
            this.w[j] += this.lr * error * xi[j];
          }
          this.b += this.lr * error;
        }
      }
    }
  }
  
  // ==============================
  // 3) Normalización simple
  // ==============================
  function normalizeDataset(X) {
    // normaliza por feature dividiendo por su max abs (min=0 en nuestros datos)
    const nFeatures = X[0].length;
    const maxValues = Array(nFeatures).fill(0);
    for (const row of X) {
      for (let j = 0; j < nFeatures; j++) {
        maxValues[j] = Math.max(maxValues[j], Math.abs(row[j]));
      }
    }
    // evitar dividir por 0
    return X.map(row => row.map((v, j) => (maxValues[j] === 0 ? v : v / maxValues[j])));
  }
  
  // ==============================
  // 4) Dataset simulado (educativo)
  // ==============================
  /*
   Features:
    - reprobadas: número de materias reprobadas (0..10)
    - asistencia: porcentaje de asistencia (0..100)
    - promedio: promedio de notas (0..20)
   Label:
    - 1 = riesgo, 0 = sin riesgo
  */
  const rawX = [
    [0, 95, 16],  // buen estudiante => 0
    [1, 90, 14],  // leve riesgo? no => 0
    [4, 60, 9],   // reprobadas >=4 y promedio bajo => 1
    [5, 50, 8],   // riesgo => 1
    [2, 75, 12],  // borderline -> 0
    [3, 65, 11],  // posible riesgo -> 1 (lo marcamos 1 para enseñar)
    [0, 85, 18],  // 0
    [6, 40, 6],   // 1
    [1, 72, 10],  // borderline -> 1 (promedio = 10)
    [0, 98, 19],  // 0
    [2, 68, 9],   // 1 (promedio bajo)
    [3, 80, 13]   // 0 (ejemplo para variabilidad)
  ];
  
  // Etiquetas (según regla simplificada)
  const rawY = rawX.map(row => {
    const reprobadas = row[0];
    const asistencia = row[1];
    const promedio = row[2];
    // regla educativa/sintética:
    return (reprobadas >= 4 || promedio < 10 || asistencia < 70) ? 1 : 0;
  });
  
  // ==============================
  // 5) Normalizar y preparar datos
  // ==============================
  const X = normalizeDataset(rawX);
  const y = rawY;
  
  // dividir en train/test (aquí usamos todo para entrenar y mostrar predicciones)
  console.log("Dataset (normalizado):");
  for (let i = 0; i < X.length; i++) {
    console.log(i, X[i], "=>", y[i]);
  }
  
  // ==============================
  // 6) Entrenar y evaluar con todas las activaciones
  // ==============================
  const activationsToTest = ["step", "sigmoid", "relu", "tanh", "linear", "softmax"];
  const epochs = 20; // >10 como pide la consigna
  const lr = 0.1;
  
  for (const act of activationsToTest) {
    console.log("\n==============================");
    console.log("Activación:", act);
    console.log("==============================");
  
    const p = new Perceptron(X[0].length, act, lr);
    p.train(X, y, epochs);
  
    // mostrar pesos aprendidos (útil para entender qué hace el perceptrón)
    console.log("Pesos:", p.w.map(w => w.toFixed(4)), "Bias:", p.b.toFixed(4));
  
    // evaluar sobre dataset de entrenamiento (por simplicidad)
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      const pred = p.predict(X[i]);
      // interpretar el output numérico: threshold 0.5 (para activaciones continuas)
      const clas = (act === "step") ? pred : (pred >= 0.5 ? 1 : 0);
      if (clas === y[i]) correct++;
      console.log(`Entrada(orig) ${rawX[i]} -> salida: ${pred.toFixed(4)} -> clase(${clas}) (esperada:${y[i]})`);
    }
    console.log(`Accuracy (sobre training): ${(correct / X.length * 100).toFixed(1)}%`);
  }
  
  