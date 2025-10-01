using System;

class Perceptron
{
    public double[] pesos;
    public double sesgo;
    public double tasaAprendizaje;
    public string activacion;

    public Perceptron(int nEntradas, double tasa = 0.1, string act = "sigmoid")
    {
        pesos = new double[nEntradas];
        sesgo = 0.0;
        tasaAprendizaje = tasa;
        activacion = act;

        // Inicializar pesos aleatoriamente
        Random rnd = new Random();
        for (int i = 0; i < pesos.Length; i++)
            pesos[i] = rnd.NextDouble() * 0.2 - 0.1;
    }

    // Funciones de activación
    public double Activar(double x)
    {
        if (activacion == "step") return x >= 0 ? 1 : 0;
        if (activacion == "sigmoid") return 1.0 / (1.0 + Math.Exp(-x));
        if (activacion == "tanh") return Math.Tanh(x);
        if (activacion == "relu") return x > 0 ? x : 0;
        if (activacion == "linear") return x;
        if (activacion == "softmax")
        {
            // Para un perceptrón binario, softmax ≈ sigmoide
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        return x;
    }

    // Predicción
    public double Predecir(double[] entrada)
    {
        double suma = sesgo;
        for (int i = 0; i < pesos.Length; i++)
            suma += pesos[i] * entrada[i];
        return Activar(suma);
    }

    // Entrenamiento
    public void Entrenar(double[][] X, double[] y, int epocas)
    {
        for (int e = 0; e < epocas; e++)
        {
            for (int i = 0; i < X.Length; i++)
            {
                double salida = Predecir(X[i]);
                double error = y[i] - salida;

                // Ajustar pesos y sesgo
                for (int j = 0; j < pesos.Length; j++)
                    pesos[j] += tasaAprendizaje * error * X[i][j];

                sesgo += tasaAprendizaje * error;
            }
        }
    }
}

class Program
{
    static void Main()
    {
        Console.WriteLine("=== Predicción del Clima con varias activaciones ===\n");

        /*
         * 🌦️ Dataset simple:
         * Entrada: [Temperatura en °C]
         * Salida: 1 = Llueve, 0 = No llueve
         */
        double[][] X = new double[][]
        {
            new double[] {10}, // frío → lluvia
            new double[] {12}, // frío → lluvia
            new double[] {28}, // calor → no lluvia
            new double[] {30}, // calor → no lluvia
            new double[] {18}, // fresco → lluvia
            new double[] {26}  // templado → no lluvia
        };

        double[] y = { 1, 1, 0, 0, 1, 0 };

        // Lista de funciones de activación a probar
        string[] activaciones = { "step", "sigmoid", "tanh", "relu", "linear", "softmax" };

        double[][] pruebas = new double[][]
        {
            new double[] {8},   // muy frío
            new double[] {14},  // fresco
            new double[] {20},  // templado
            new double[] {27},  // caluroso
            new double[] {32}   // muy caluroso
        };

        // Probar todas las activaciones
        foreach (string act in activaciones)
        {
            Console.WriteLine($"\n--- Activación: {act} ---");
            Perceptron p = new Perceptron(1, 0.1, act);
            p.Entrenar(X, y, 20); // más de 10 iteraciones

            foreach (var temp in pruebas)
            {
                double salida = p.Predecir(temp);
                Console.WriteLine($"Temp {temp[0]}°C → {(salida >= 0.5 ? "Lluvia" : "No Lluvia")} (Pred: {salida:F2})");
            }
        }
    }
}
