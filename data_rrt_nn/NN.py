import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import re

# sample points of path
NUM_POINTS = 100
CORNER_TYPE_MAP = {"corner": 0, "default": 1}  

def load_and_process_data(file_list):
    """Load and process all training data files"""
    X_train = []  
    Y_train = []  
    
    x_means = []
    x_stds = []
    y_means = []
    y_stds = []
    
    for filename in file_list:
        if not os.path.isfile(filename):
            print(f"Warning: File '{filename}' not found. Skipping.")
            continue
        
        print(f"Processing file: {filename}")
        
        velocity = extract_velocity_from_filename(filename)
        corner_type = "corner" if "corner" in filename.lower() else "default"
        corner_type_encoded = CORNER_TYPE_MAP[corner_type]
        
        x, y = load_route(filename)
        x_resampled, y_resampled = resample_route(x, y, NUM_POINTS)
        
        x_mean, x_std = np.mean(x_resampled), np.std(x_resampled)
        y_mean, y_std = np.mean(y_resampled), np.std(y_resampled)
        
        x_means.append(x_mean)
        x_stds.append(x_std)
        y_means.append(y_mean)
        y_stds.append(y_std)
        
        x_norm = (x_resampled - x_mean) / x_std
        y_norm = (y_resampled - y_mean) / y_std
        
        X_train.append([velocity, corner_type_encoded])
        
        path = np.column_stack((x_norm, y_norm)).flatten()
        Y_train.append(path)
        
        # Visualize the loaded path
        plt.figure(figsize=(8, 6))
        plt.plot(x_resampled, y_resampled, 'b-', linewidth=2)
        plt.plot(x_resampled[0], y_resampled[0], 'go', markersize=8, label='Start')
        plt.plot(x_resampled[-1], y_resampled[-1], 'ro', markersize=8, label='End')
        plt.title(f"Path from {filename}")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

    norm_params = {
        'x_mean': np.mean(x_means),
        'x_std': np.mean(x_stds),
        'y_mean': np.mean(y_means),
        'y_std': np.mean(y_stds)
    }
    
    return np.array(X_train), np.array(Y_train), norm_params

def load_route(filename):
    """Load route data from file"""
    try:
        data = np.loadtxt(filename, delimiter=',')
        x = data[:, 1]  
        y = data[:, 2]  
        return x, y
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return np.linspace(0, 10, 50), np.linspace(0, 10, 50)

def extract_velocity_from_filename(filename):
    """Extract velocity value from filename"""
    match = re.search(r'position_([\d\.]+)_', filename)
    if match:
        return float(match.group(1))
    else:
        print(f"Could not extract velocity from {filename}, using default 3.0")
        return 3.0

def resample_route(x, y, num_points):
    """Resample route to have a fixed number of points"""
    dx = np.diff(x)
    dy = np.diff(y)
    segments = np.sqrt(dx**2 + dy**2)
    if len(segments) == 0:
        return x, y  
    
    cumulative_dist = np.concatenate(([0], np.cumsum(segments)))
    
    uniform_dist = np.linspace(0, cumulative_dist[-1], num_points)
    
    x_uniform = np.interp(uniform_dist, cumulative_dist, x)
    y_uniform = np.interp(uniform_dist, cumulative_dist, y)
    
    return x_uniform, y_uniform

def build_model(input_dim, output_dim):
    """Build a neural network model"""
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dropout(0.2),  
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dense(output_dim, activation='linear') 
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'  
    )
    
    return model

def train_model(X_train, Y_train):
    """Train the neural network model"""

    input_dim = X_train.shape[1]  
    output_dim = Y_train.shape[1]  
    
    print(f"Building model with input_dim={input_dim}, output_dim={output_dim}")
    
    model = build_model(input_dim, output_dim)
    model.summary()
    
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=50,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, Y_train,
        epochs=300,
        batch_size=4,  
        callbacks=[early_stopping],
        verbose=1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    try:
        save_model(model, "cornering_path_model.keras")
        print("Model saved as 'cornering_path_model.keras'")
    except Exception as e:
        print(f"Failed to save model: {e}")
    
    return model

def predict_path(model, velocity, corner_type, norm_params):
    """Predict a path based on velocity and corner type"""
    corner_type_encoded = CORNER_TYPE_MAP.get(corner_type.lower(), 0)
    input_data = np.array([[velocity, corner_type_encoded]])
    
    print(f"Predicting path for velocity={velocity}, corner_type={corner_type}")
    
    prediction = model.predict(input_data)[0]
    
    path = prediction.reshape(-1, 2)
    
    x_denorm = path[:, 0] * norm_params['x_std'] + norm_params['x_mean']
    y_denorm = path[:, 1] * norm_params['y_std'] + norm_params['y_mean']
    
    return np.column_stack((x_denorm, y_denorm))

def plot_path(path, title="Predicted Path"):
    """Plot a path"""
    plt.figure(figsize=(10, 8))
    plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2)
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')  
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, label='End')  
    
    num_arrows = 5
    indices = np.linspace(0, len(path)-2, num_arrows, dtype=int)
    for i in indices:
        plt.arrow(
            path[i, 0], path[i, 1],
            path[i+1, 0] - path[i, 0], path[i+1, 1] - path[i, 1],
            head_width=0.5, head_length=0.7, fc='k', ec='k'
        )
    
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def compare_all_paths(model, norm_params):
    """Generate and compare paths for all velocity/corner combinations"""
    velocities = [2.5, 3.0, 3.5, 4.0]
    corner_types = ["corner", "default"]
    
    plt.figure(figsize=(15, 10))
    
    for corner_type in corner_types:
        for velocity in velocities:
            path = predict_path(model, velocity, corner_type, norm_params)
            plt.plot(
                path[:, 0], path[:, 1], 
                label=f"{corner_type.capitalize()} - {velocity} velocity"
            )
    
    plt.title("Comparison of All Predicted Paths")
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def main():
    file_list = [
        "position_2.5_corner.txt",
        "position_2.5_default.txt",
        "position_3.0_corner.txt",
        "position_3.0_default.txt",
        "position_3.5_corner.txt",
        "position_3.5_default.txt",
        "position_4.0_corner.txt"
    ]
    
    print("Loading and processing data...")
    X_train, Y_train, norm_params = load_and_process_data(file_list)
    print(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")

    if X_train.shape[0] < 2:
        print("Not enough valid data files found for training.")
        return
    
    model_path = "cornering_path_model.keras"
    if os.path.exists(model_path):
        try:
            print(f"Loading existing model from {model_path}")
            model = load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Training new model...")
            model = train_model(X_train, Y_train)
    else:
        print("Training new model...")
        model = train_model(X_train, Y_train)
    
    print("Generating comparison of all predicted paths...")
    compare_all_paths(model, norm_params)
    
    while True:
        print("\n===== Cornering Path Prediction =====")
        corner_type = input("Enter corner type (corner/default): ").strip().lower()
        
        if corner_type not in ["corner", "default"]:
            print("Invalid corner type. Using 'corner' as default.")
            corner_type = "corner"
        
        try:
            velocity = float(input("Enter velocity (2.5-4.0): ").strip())
            if velocity < 2.5 or velocity > 4.0:
                print("Velocity out of training range (2.5-4.0). Results may be less accurate.")
        except ValueError:
            print("Invalid velocity. Using 3.0 as default.")
            velocity = 3.0
        

        predicted_path = predict_path(model, velocity, corner_type, norm_params)
        plot_path(
            predicted_path,
            f"Predicted Path for {corner_type.capitalize()} at {velocity} velocity"
        )
        
        if input("\nPredict another path? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()