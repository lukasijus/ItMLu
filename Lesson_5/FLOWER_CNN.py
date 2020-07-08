# Flower CNN
import sys
import os
sys.path.append(os.path.abspath('..'))
import shutil
import glob
import initialize_imports as imp

# zip_dir = imp.download('https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', 'flower_photos.tgz')
#
data_directory = 'C:\\Users\\lukas.rimkus\\.keras\\datasets\\flower_photos'

print('data directory: ', data_directory)


train_daisy_dir = os.path.join(data_directory,      'train/daisy')
train_roses_dir = os.path.join(data_directory,      'train/roses')
train_sunflowers_dir = os.path.join(data_directory, 'train/sunflowers')
train_tulips_dir = os.path.join(data_directory,     'train/tulips')
train_dandelion_dir = os.path.join(data_directory,  'train/dandelion')

val_daisy_dir = os.path.join(data_directory,      'validation\\daisy')
val_roses_dir = os.path.join(data_directory,      'validation\\roses')
val_sunflowers_dir = os.path.join(data_directory, 'validation\\sunflowers')
val_tulips_dir = os.path.join(data_directory,     'validation\\tulips')
val_dandelion_dir = os.path.join(data_directory,  'validation\\dandelion')

train_num_daisy = len(os.listdir(       train_daisy_dir))
train_num_roses = len(os.listdir(       train_roses_dir))
train_num_sunflowers = len(os.listdir(  train_sunflowers_dir))
train_num_tulips = len(os.listdir(      train_tulips_dir))
train_num_dandelion = len(os.listdir(   train_dandelion_dir))

val_num_daisy = len(os.listdir(       val_daisy_dir))
val_num_roses = len(os.listdir(       val_roses_dir))
val_num_sunflowers = len(os.listdir(  val_sunflowers_dir))
val_num_tulips = len(os.listdir(      val_tulips_dir))
val_num_dandelion = len(os.listdir(   val_dandelion_dir))

print('train number of daisy: ',      train_num_daisy)
print('train number of sunflowers: ', train_num_sunflowers)
print('train number of tulips: ',     train_num_tulips)
print('train number of roses: ',      train_num_roses)
print('train number of candelion: ',  train_num_dandelion)

print('validation number of daisy: ',      val_num_daisy)
print('validation number of sunflowers: ', val_num_sunflowers)
print('validation number of tulips: ',     val_num_tulips)
print('validation number of roses: ',      val_num_roses)
print('validation number of candelion: ',  val_num_dandelion)

train_num = train_num_daisy + \
            train_num_roses + \
            train_num_tulips + \
            train_num_dandelion + \
            train_num_sunflowers

val_num =   val_num_daisy + \
            val_num_roses + \
            val_num_tulips + \
            val_num_dandelion + \
            val_num_sunflowers

print('Training Examples: ', train_num)
print('Validation Examples: ', val_num)

train_dir = os.path.join(data_directory, 'train')
val_dir = os.path.join(data_directory, 'validation')

BATCH_SIZE = 120
IMAGE_SHAPE = 150

# DATA AUGMENTATION
train_data_params = imp.tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    zoom_range=0.5,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
                        )
# PREPARE DATA FOR THE MODEL
train_data = train_data_params.flow_from_directory(directory=train_dir, batch_size=BATCH_SIZE, shuffle=True, target_size=(IMAGE_SHAPE,IMAGE_SHAPE), class_mode='sparse')
validation_data = train_data_params.flow_from_directory(directory=val_dir, batch_size=BATCH_SIZE, shuffle=False, target_size=(IMAGE_SHAPE,IMAGE_SHAPE), class_mode='sparse' )
# Build a model
model = imp.tf.keras.models.Sequential([
    imp.tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape= (IMAGE_SHAPE, IMAGE_SHAPE, 3)),
    imp.tf.keras.layers.MaxPooling2D(2, 2),

    imp.tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    imp.tf.keras.layers.MaxPooling2D(2, 2),

    imp.tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    imp.tf.keras.layers.MaxPooling2D(2, 2),

    imp.tf.keras.layers.Flatten(),
    imp.tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    imp.tf.keras.layers.MaxPooling2D(2, 2),

    
    imp.tf.keras.layers.Flatten(),
    imp.tf.keras.layers.Dense(512, activation='relu'),
    imp.tf.keras.layers.Dense(5, activation='softmax')
])

# compile a model
model.compile(optimizer='adam', loss='sparse_crossentropy',metrics=['accuracy'])

model.summary()
# Train a model
history = model.fit(train_data, epochs=80, steps_per_epoch= imp.np.ceil(train_num/BATCH_SIZE), validation_data=validation_data,validation_steps=imp.np.ceil(val_num/BATCH_SIZE))


#
# Visualizing
# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
imp.plt.plot(epochs, acc)
imp.plt.plot(epochs, val_acc)
imp.plt.legend((acc, val_acc), ('acc', 'val_acc'))
imp.plt.title('Training and validation accuracy')

imp.plt.figure()

# Plot training and validation loss per epoch
imp.plt.plot(epochs, loss)
imp.plt.plot(epochs, val_loss)
imp.plt.legend((loss, val_loss), ('loss', 'val_loss'))
imp.plt.title('Training and validation `loss')
imp.plt.show()




































