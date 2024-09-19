from keras import models
from keras import layers
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import fashion_mnist


class LoadData:
    def __init__(self):
        # Laden der Dateien
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = fashion_mnist.load_data()

    def prepare_data(self):
        """
        Bereitet die Bilddaten vor: Reshape und Normalisierung.
        """
        self.X_train = self.X_train.reshape((60000, 784))
        self.X_test = self.X_test.reshape((10000, 784))

        # Normalisierung
        self.X_train = self.X_train.astype("float32") / 255
        self.X_test = self.X_test.astype("float32") / 255

        # One-Hot-Encoding der Labels
        self.Y_train = to_categorical(self.Y_train)
        self.Y_test = to_categorical(self.Y_test)


class DisplayDataShape:
    def __init__(self, viewer):
        self.viewer = viewer

    def show_data_shapes(self):
        """
        Anzeige der Datenformen (Trainings- und Testdaten) in Streamlit.
        """
        X_train_shape, Y_train_shape = self.viewer.X_train.shape, self.viewer.Y_train.shape
        X_test_shape, Y_test_shape = self.viewer.X_test.shape, self.viewer.Y_test.shape
        st.header("Daten-Formen")
        st.write(f"Trainingsdaten: {X_train_shape}")
        st.write(f"Trainingslabels: {Y_train_shape}")
        st.write(f"Testdaten: {X_test_shape}")
        st.write(f"Testlabels: {Y_test_shape}")


class ExampleImageDisplay:
    def __init__(self, viewer):
        self.viewer = viewer

    def display_image(self):
        """
        Zeigt ein Beispielbild aus den Trainingsdaten an, basierend auf dem ausgewählten Index.
        """
        st.header("Beispiel-Bilder")
        index = st.number_input("Bitte gib einen Index für ein Beispiel-Bild ein", min_value=0, max_value=59999, step=1, value=0)

        digit = self.viewer.X_train[index].reshape(28, 28)  # Rückverwandeln in 28x28 für die Anzeige
        fig, ax = plt.subplots()  # Erstellt eine neue Figur für das Beispielbild
        ax.imshow(digit, cmap=plt.cm.binary)
        st.pyplot(fig)  # Beispielbild anzeigen
        st.write(f"Label: {np.argmax(self.viewer.Y_train[index])}")


class NeuralNetworkBuilder:
    def __init__(self, viewer):
        self.viewer = viewer
        self.model = None
        self.history = None  

    def configure_and_train(self):
        """
        Konfigurieren, Erstellen und Trainieren des neuronalen Netzwerks.
        """
        st.sidebar.header("Konfiguration des Neuronalen Netzwerks")

        # Anzahl der Hidden Layers
        hidden_layers = st.sidebar.slider("Anzahl der Hidden Layers:", 1, 10, 5)

        # Anzahl der Neuronen pro Layer
        neurons = st.sidebar.selectbox("Anzahl der Neuronen je Layer", [32, 64, 128, 256, 512, 1024])

        # Anzahl der Epochen
        epochs = st.sidebar.number_input("Anzahl der Epochen", min_value=1, max_value=50, value=10)

        # Button zur Ausführung des Trainings
        if st.sidebar.button("Modell erstellen und trainieren"):
            # Modell erstellen
            self.model = models.Sequential([layers.InputLayer(input_shape=(784,))])
            for i in range(hidden_layers):
                self.model.add(layers.Dense(neurons, activation='sigmoid', name=f"hidden_layer_{i + 1}"))
            self.model.add(layers.Dense(10, activation='sigmoid', name="output_layer"))

            st.success(f"Modell mit {hidden_layers} Hidden Layers und {neurons} Neuronen pro Layer erstellt.")

            # Modell kompilieren
            self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
            st.success("Modell erfolgreich kompiliert.")

            # Modell trainieren
            st.info("Training wird gestartet...")
            progress_bar = st.progress(0)
            epoch_message = st.empty()  # Ein leerer Bereich zum Aktualisieren des Fortschritts jeder Epoche

            # Training und speichern der Historie
            self.history = self.model.fit(
                self.viewer.X_train,
                self.viewer.Y_train,
                epochs=epochs,
                shuffle=True,
                validation_split=0.1,
                verbose=1 
            )

            for epoch in range(epochs):
                progress_bar.progress((epoch + 1) / epochs)
                epoch_message.text(f"Epoche {epoch + 1}/{epochs} abgeschlossen.")

            st.success(f"Training abgeschlossen nach {epochs} Epochen.")

            # Plot der Trainingsgenauigkeit vs Validierungsgenauigkeit
            self.plot_training_accuracy()

            # Netzwerk evaluieren
            self.evaluate_model()

            # Confusion Matrix anzeigen
            self.plot_confusion_matrix()

            # Vorhersagen für ein Bild
            self.classify_image()

    def plot_training_accuracy(self):
        """
        Plot der Trainingsgenauigkeit vs. Validierungsgenauigkeit.
        """
        st.header("Trainingsgenauigkeit vs. Validierungsgenauigkeit")
        fig, ax = plt.subplots() 
        ax.plot(self.history.history["accuracy"], label="Training_Acc")
        ax.plot(self.history.history["val_accuracy"], label="Validation_Acc")
        ax.legend()
        ax.set_xlabel("Epochen")
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)  

    def evaluate_model(self):
        """
        Evaluierung des Modells nach dem Training.
        """
        loss_val, acc = self.model.evaluate(self.viewer.X_test, self.viewer.Y_test, verbose=0)
        st.write(f"Loss: {loss_val}")
        st.write(f"Accuracy: {acc}")

    def plot_confusion_matrix(self):
        """
        Zeigt die Confusion Matrix basierend auf den Vorhersagen.
        """
        predictions = self.model.predict(self.viewer.X_test)
        rounded_predictions = np.argmax(predictions, axis=1)
        cm = confusion_matrix(np.argmax(self.viewer.Y_test, axis=1), rounded_predictions)
        fig, ax = plt.subplots() 
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
        disp.plot(cmap='Blues', values_format='d', ax=ax)
        plt.title('Confusion Matrix')
        st.pyplot(fig)  

    def classify_image(self):
        """
        Klassifiziert ein einzelnes Bild aus dem Testset basierend auf dem vom Benutzer angegebenen Index.
        """
        st.header("Bildklassifikation")
        index = st.number_input("Bildindex zur Klassifizierung auswählen", min_value=0, max_value=9999, step=1, value=0)

        # Vorhersage
        prediction = self.model.predict(self.viewer.X_test[index].reshape(1, 784))
        rounded_prediction = np.argmax(prediction)

        st.write(f"Echtes Label: {np.argmax(self.viewer.Y_test[index])}")
        st.write(f"Vorhergesagtes Label: {rounded_prediction}")


class StreamlitApp:
    def __init__(self, viewer):
        self.viewer = viewer
        self.model = None

        # Display-Optionen
        self.display_classes = {
            "Daten-Form anzeigen": DisplayDataShape(viewer),
            "Beispiel-Bilder anzeigen": ExampleImageDisplay(viewer)
        }

    def run(self):
        """
        Hauptsteuerung der Anzeige und Netzwerkkonfiguration.
        """
        # Menü 1: Datenanzeige
        st.sidebar.header("Datenanzeige")
        data_display_option = st.sidebar.selectbox("Wähle eine Funktion", ["Daten-Form anzeigen", "Beispiel-Bilder anzeigen"])

        # Daten vorbereiten
        self.viewer.prepare_data()

        # Ausführung der Anzeige-Funktion
        if data_display_option == "Daten-Form anzeigen":
            self.display_classes["Daten-Form anzeigen"].show_data_shapes()
        elif data_display_option == "Beispiel-Bilder anzeigen":
            self.display_classes["Beispiel-Bilder anzeigen"].display_image()

        # Neuronales Netzwerk konfigurieren und trainieren
        network_builder = NeuralNetworkBuilder(self.viewer)
        network_builder.configure_and_train()


def main():
    viewer = LoadData()
    app = StreamlitApp(viewer)
    app.run()


if __name__ == "__main__":
    main()
