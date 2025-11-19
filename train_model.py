import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

test_all_versions = False  
dataset_index = 8

def train_on_dataset(index):
    print(f"\n==============================")
    print(f" Training on data{index}.pickle")
    print(f"==============================")

    with open(f"data{index}.pickle", "rb") as f:
        data = pickle.load(f)

    x_train, y_train = data['x_train'], data['y_train']
    x_valid, y_valid = data['x_validation'], data['y_validation']
    labels = data.get('labels', [str(i) for i in range(len(set(y_train)))])

    print("Train shape:", x_train.shape)
    print("Validation shape:", x_valid.shape)
    print("Number of classes:", len(set(y_train)))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_valid = x_valid.transpose(0, 2, 3, 1)

    input_shape = (32, 32, 1) if x_train.shape[-1] == 1 else (32, 32, 3)

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0

    num_classes = len(set(y_train))

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)

    history = model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    val_loss, val_acc = model.evaluate(x_valid, y_valid, verbose=0)
    print(f"data{index} → Final Validation Accuracy: {val_acc*100:.2f}%")

    model.save(f"traffic_sign_model_data{index}.h5")
    print(f"Model saved as traffic_sign_model_data{index}.h5")

    return val_acc


if test_all_versions:
    results = {}
    for i in range(9):
        acc = train_on_dataset(i)
        results[f"data{i}"] = acc

    print("\n📊 Summary of all datasets:")
    for k, v in results.items():
        print(f"{k}: {v*100:.2f}%")

else:
    train_on_dataset(dataset_index)
