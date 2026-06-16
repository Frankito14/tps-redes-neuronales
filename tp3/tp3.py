import numpy as np
import tensorflow as tf
from keras import Input, Model, metrics, losses, optimizers
from keras import layers
import matplotlib.pyplot as plt

# Capa personalizada para el truco de reparametrización
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# CAMBIO: Subimos la dimensión latente para capturar más detalles (ej. 64)
latent_dim = 64

# ================= ENCODER =================
encoder_inputs = Input(shape=(64, 64, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)

# CAMBIO: Aumentamos esta capa a 256 para evitar un cuello de botella antes del espacio latente
x = layers.Dense(256, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# ================= DECODER =================
latent_inputs = Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 128))(x)

x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

decoder = Model(latent_inputs, decoder_outputs, name="decoder")

# ================= VAE MODEL =================
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# 1. Instanciar y compilar
vae = VAE(encoder, decoder)
vae.compile(optimizer=optimizers.Adam(learning_rate=0.0005))


# ================= NUEVAS FUNCIONES DE VISUALIZACIÓN =================

# Función 1: Compara Pokémon reales del dataset contra su reconstrucción
def plot_reconstructions(vae, dataset, n=8):
    for batch in dataset.take(1):
        images = batch[:n]
        z_mean, _, _ = vae.encoder.predict(images, verbose=0)
        reconstructed = vae.decoder.predict(z_mean, verbose=0)
        
        plt.figure(figsize=(n * 2, 4))
        for i in range(n):
            # Mostrar Originales
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].numpy().reshape(64, 64), cmap="Greys_r")
            plt.axis("off")
            if i == 0: ax.set_title("Originales", fontsize=10, weight='bold')
            
            # Mostrar Reconstruidos
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed[i].reshape(64, 64), cmap="Greys_r")
            plt.axis("off")
            if i == 0: ax.set_title("Reconstruidos", fontsize=10, weight='bold')
        plt.tight_layout()
        plt.show()

# Función 2: Genera Pokémon aleatorios totalmente nuevos desde el espacio 64D
def generate_random_pokemon(decoder, latent_dim, n=8):
    # Creamos vectores aleatorios usando una distribución normal estándar
    random_latent_vectors = np.random.normal(size=(n, latent_dim))
    generated_images = decoder.predict(random_latent_vectors, verbose=0)
    
    plt.figure(figsize=(n * 2, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(generated_images[i].reshape(64, 64), cmap="Greys_r")
        plt.axis("off")
    plt.suptitle("Pokémon Inventados (Muestreo Aleatorio)", y=1.05, weight='bold')
    plt.tight_layout()
    plt.show()


# ================= CARGA DE DATOS Y ENTRENAMIENTO =================

ruta_dataset = "Sprites_BN" 

print("Cargando imágenes...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    ruta_dataset,
    label_mode=None,          
    color_mode="grayscale",   
    image_size=(64, 64),      
    batch_size=32,            
    shuffle=True              
)

# Normalizar los píxeles a [0, 1]
train_dataset = train_dataset.map(lambda x: x / 255.0)

print("¡Iniciando el entrenamiento!")
epochs = 100 
history = vae.fit(train_dataset, epochs=epochs)

# Al finalizar el entrenamiento, ejecutamos los nuevos métodos de testeo
print("Entrenamiento finalizado. Evaluando resultados...")
print("\n--- TEST 1: Calidad de Reconstrucción ---")
plot_reconstructions(vae, train_dataset)

print("\n--- TEST 2: Generación de Nuevos Sprites ---")
generate_random_pokemon(vae.decoder, latent_dim)