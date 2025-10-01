import java.util.*;

class Perceptron {
    double[] weights;
    double bias;
    double lr;
    String activation;

    Perceptron(int inputSize, String activation, double lr) {
        Random rnd = new Random();
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) weights[i] = rnd.nextDouble() * 2 - 1;
        bias = rnd.nextDouble() * 2 - 1;
        this.lr = lr;
        this.activation = activation;
    }

    double activate(double x) {
        switch (activation) {
            case "linear": return x;
            case "step": return x >= 0 ? 1 : 0;
            case "sigmoid": return 1.0 / (1.0 + Math.exp(-x));
            case "relu": return Math.max(0, x);
            case "tanh": return Math.tanh(x);
            default: return x;
        }
    }

    double predict(double[] inputs) {
        double total = 0;
        for (int i = 0; i < inputs.length; i++) total += inputs[i] * weights[i];
        total += bias;
        return activate(total);
    }

    void train(List<double[]> X, List<Double> y, int epochs) {
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < X.size(); i++) {
                double pred = predict(X.get(i));
                double error = y.get(i) - pred;
                for (int j = 0; j < weights.length; j++)
                    weights[j] += lr * error * X.get(i)[j];
                bias += lr * error;
            }
        }
    }
}

public class Main {
    static void runCase(String name, String description, List<double[]> X, List<Double> y) {
        Perceptron p = new Perceptron(X.get(0).length, "step", 0.1);
        p.train(X, y, 15);

        System.out.println("\n=== Caso: " + name + " ===");
        System.out.println("Descripción: " + description);
        System.out.println("Entradas -> Salida esperada | Predicción");
        for (int i = 0; i < X.size(); i++) {
            System.out.println(Arrays.toString(X.get(i)) + " -> " + y.get(i).intValue() + " | " + (int)p.predict(X.get(i)));
        }
    }

    public static void main(String[] args) {
        System.out.println("=== Perceptrón Simple en Java ===");
        System.out.println("Función de activación usada: STEP (escalón)");
        System.out.println("Entrenamiento en diferentes casos prácticos.\n");

        // 1. AND lógico
        runCase("AND",
            "La salida es 1 SOLO cuando ambas entradas son 1.",
            Arrays.asList(new double[]{0,0}, new double[]{0,1}, new double[]{1,0}, new double[]{1,1}),
            Arrays.asList(0.0,0.0,0.0,1.0));

        // 2. OR lógico
        runCase("OR",
            "La salida es 1 si al menos una de las entradas es 1.",
            Arrays.asList(new double[]{0,0}, new double[]{0,1}, new double[]{1,0}, new double[]{1,1}),
            Arrays.asList(0.0,1.0,1.0,1.0));

        // 3. Spam/No Spam
        runCase("Spam",
            "Ejemplo simple: si el correo tiene ciertas palabras clave, es SPAM (1), de lo contrario No Spam (0).",
            Arrays.asList(new double[]{1,0}, new double[]{0,1}, new double[]{1,1}, new double[]{0,0}),
            Arrays.asList(1.0,1.0,1.0,0.0));

        // 4. Clima
        runCase("Clima",
            "Valores altos = día soleado (1), valores bajos = día lluvioso (0).",
            Arrays.asList(new double[]{30}, new double[]{10}, new double[]{20}, new double[]{35}),
            Arrays.asList(1.0,0.0,0.0,1.0));

        // 5. Fraude
        runCase("Fraude",
            "Transacciones muy altas pueden indicar fraude (1), montos bajos = normales (0).",
            Arrays.asList(new double[]{1000}, new double[]{50}, new double[]{2000}, new double[]{10}),
            Arrays.asList(1.0,0.0,1.0,0.0));

        // 6. Riesgo académico
        runCase("Riesgo académico",
            "Pocas horas de estudio = riesgo (1), muchas horas = bajo riesgo (0).",
            Arrays.asList(new double[]{8}, new double[]{4}, new double[]{6}, new double[]{2}),
            Arrays.asList(0.0,1.0,0.0,1.0));
    }
}
