"""
This example demonstrates how to create, and train a hydraulic surrogate model.
"""
from epyt_control.models import PIGNNModel


if __name__ == "__main__":
    # Create a new hydrauic surrogate model for Anytown
    model = PIGNNModel.from_network("anytown")
    #model = PIGNNModel("Anytown.inp")  # Alternatively, you can load an arbitrary .inp file via the constructor

    # Generate test data and training demands from the .inp file
    model.load_from_inp()
    model.prepare_data(train_ratio=0.6, val_ratio=0.2)

    # Build the neural network
    model.build_model()

    # Train the model and store it in a file
    model.train()
    model.save_model(f"anytown_hydsurrogate_pignn.pt")

    # Evaluate its performance on the test data -- i.e.,
    # compare predicted heads and flows to EPANET simulation
    pred = model.evaluate()
