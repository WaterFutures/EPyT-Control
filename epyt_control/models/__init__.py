try:
    from .pi_gnn_hydsurrogate import PIGNNModel, device
except:
    print("Failed to import 'PIGNNModel'.")
    print("You probably did not correctly install the optional extension [hydsurrogate] " +
          "-- please run 'pip install epyt-control[hydsurrogate]'")
