try:
    from .pi_gnn_hydsurrogate import PIGNNModel, device
except:
    print("Failed to import 'PIGNNModel'.")
    print("You probably did not correctly install the optional extension [hydsurrogate] " +
          "-- please run 'pip install epyt-control[hydsurrogate]'")

try:
    from .qualsurrogate_mega_mp import *
except:
    print("Failed to import 'qualsurrogate_mega_mp'.")
    print("You probably did not correctly install the optional extension [qualsurrogate] " +
          "-- please run 'pip install epyt-control[qualsurrogate]'")
