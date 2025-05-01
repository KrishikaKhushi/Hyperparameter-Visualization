import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Image dimensions
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

# Size of the noise vector used as generator input
z_dim = 100

def build_generator():
    inputs = Input(shape=(z_dim,))
    
    # First dense layer
    x = Dense(7 * 7 * 256)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Reshape((7, 7, 256))(x)
    
    # Upsample to 14x14
    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Upsample to 28x28
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Output layer with tanh activation
    outputs = Conv2D(channels, kernel_size=3, padding='same', activation='tanh')(x)
    
    model = Model(inputs, outputs, name="generator")
    return model

def build_discriminator():
    inputs = Input(shape=img_shape)
    
    # First convolutional layer
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Second convolutional layer
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    
    # Output layer
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="discriminator")
    return model

def load_circle_dataset(dataset_path):
    images = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_path, filename)
            img = load_img(img_path, color_mode='grayscale', target_size=(img_rows, img_cols))
            img_array = img_to_array(img)
            images.append(img_array)
    
    # Convert to numpy array and normalize
    images = np.array(images)
    # Normalize images to [-1, 1] (to match tanh activation in generator)
    images = (images.astype(np.float32) - 127.5) / 127.5
    
    return images

# Binary cross entropy loss
def binary_cross_entropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

class CircleGAN:
    def __init__(self, optimizer_name, learning_rate=0.0002):
        # Build models
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        
        # Set optimizers based on input
        if optimizer_name.lower() == 'adam':
            # Good optimizer - should produce clear results
            self.g_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5)
            self.d_optimizer = Adam(learning_rate=learning_rate, beta_1=0.5)
        elif optimizer_name.lower() == 'rmsprop':
            # Decent optimizer - should produce okay results
            self.g_optimizer = RMSprop(learning_rate=learning_rate)
            self.d_optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            # Basic optimizer with high learning rate - likely to produce blurry results
            self.g_optimizer = SGD(learning_rate=learning_rate*5, momentum=0.0)
            self.d_optimizer = SGD(learning_rate=learning_rate*5, momentum=0.0)
        
        self.optimizer_name = optimizer_name
    
    @tf.function
    def train_discriminator_step(self, real_images, batch_size):
        # Generate noise
        noise = tf.random.normal([batch_size, z_dim])
        
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images = self.generator(noise, training=True)
            
            # Get discriminator outputs for real and fake images
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            # Calculate discriminator loss
            real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
            d_loss = real_loss + fake_loss
        
        # Compute gradients and update weights
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Calculate and return metrics
        d_accuracy = 0.5 * (tf.reduce_mean(tf.cast(real_output > 0.5, tf.float32)) + 
                          tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32)))
        
        return d_loss, d_accuracy
    
    @tf.function
    def train_generator_step(self, batch_size):
        # Generate noise
        noise = tf.random.normal([batch_size, z_dim])
        
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_images = self.generator(noise, training=True)
            
            # Get discriminator output for fake images
            fake_output = self.discriminator(fake_images, training=True)
            
            # Calculate generator loss (want discriminator to think fake images are real)
            g_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)
        
        # Compute gradients and update weights
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return g_loss
    
    def train(self, dataset, epochs=5000, batch_size=32, save_interval=500):
        # Create output directory
        output_dir = f'output_{self.optimizer_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure batch_size is not larger than dataset
        batch_size = min(batch_size, len(dataset))
        
        for epoch in range(epochs):
            # Select a random batch of real images
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            real_imgs = dataset[idx]
            
            # Train discriminator
            d_loss, d_accuracy = self.train_discriminator_step(real_imgs, batch_size)
            
            # Train generator
            g_loss = self.train_generator_step(batch_size)
            
            # Print progress
            if epoch % 100 == 0:
                print(f"{self.optimizer_name} - Epoch {epoch}/{epochs} [D loss: {d_loss:.4f}, acc.: {d_accuracy * 100:.2f}%] [G loss: {g_loss:.4f}]")
            
            # Save generated images at specified intervals
            if epoch % save_interval == 0:
                self.save_imgs(epoch, output_dir)
        
        # Save the final model
        self.generator.save(f'circle_generator_{self.optimizer_name}.h5')
        print(f"Training complete for {self.optimizer_name}. Model saved.")
        
        # Generate a final image and return it
        return self.generate_image()
    
    def save_imgs(self, epoch, output_dir, examples=16, dim=(4, 4), figsize=(10, 10)):
        # Generate random noise
        noise = tf.random.normal([examples, z_dim])
        
        # Generate images
        gen_imgs = self.generator(noise, training=False).numpy()
        
        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        # Create figure
        fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
        
        # Plot each generated image
        cnt = 0
        for i in range(dim[0]):
            for j in range(dim[1]):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        
        plt.suptitle(f"Generated Images with {self.optimizer_name} - Epoch {epoch}")
        plt.tight_layout()
        
        # Save figure
        fig.savefig(f"{output_dir}/circles_epoch_{epoch}.png")
        plt.close()
    
    def generate_image(self):
        # Generate a single image
        noise = tf.random.normal([1, z_dim])
        gen_img = self.generator(noise, training=False).numpy()
        
        # Rescale from [-1, 1] to [0, 1]
        gen_img = 0.5 * gen_img + 0.5
        
        return gen_img[0, :, :, 0]

def display_comparison(images, titles):
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each image
    for i in range(3):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(f"{titles[i]}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("optimizer_comparison.png")
    plt.show()

def main():
    # Load dataset
    dataset = load_circle_dataset('circles')
    
    if len(dataset) == 0:
        print("No images found in the 'circles' directory. Please check your dataset.")
        return
    
    print(f"Loaded dataset with {len(dataset)} images.")
    
    # Train with three different optimizers
    optimizers = ['Adam', 'RMSprop', 'SGD']
    results = []
    
    for optimizer in optimizers:
        print(f"\nTraining with {optimizer} optimizer...")
        gan = CircleGAN(optimizer)
        
        # Use fewer epochs to speed up demonstration
        result = gan.train(dataset, epochs=10000, batch_size=32, save_interval=500)
        results.append(result)
    
    # Display results side by side
    display_comparison(results, optimizers)

if __name__ == "__main__":
    main()