from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_nn(input_dim, units=64, dropout=0.3):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(units//2, activation='relu'),
        Dropout(dropout/2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
